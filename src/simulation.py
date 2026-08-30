"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import os
import random
import logging
from typing import List, Tuple, Dict, Set, Optional
from agent import Agent
from ollama_client import OllamaClient
from utils import get_place_at_position

logger = logging.getLogger(__name__)

# Constants
MAX_POSITION_ATTEMPTS = 1000
LOG_INTERVAL = 10


class Simulation:
    """Main simulation class for LLM-based agent in 2D worlds with multiple places."""
    
    def __init__(self, config: Dict, output_dir: Optional[str] = None):
        """Initialize simulation from an already parsed configuration

        Takes the parsed dictionary rather than a path so the YAML file is read
        exactly once per run (see utils.load_config).
        """
        self.config = config

        # Output directory for logs
        self.output_dir = output_dir

        # Counterfactual twin runs must start from the same world, so the seed
        # covers initial positions, personas and budgets. Upstream leaves this
        # unset and is deliberately irreproducible; omitting it keeps that.
        self.seed = self.config.get('seed')
        if self.seed is not None:
            random.seed(self.seed)
            logger.info(f"Random seed fixed: {self.seed}")
        
        # Simulation parameters
        sim_config = self.config['simulation']
        self.duration = sim_config['duration']
        self.half_space_size = sim_config['half_space_size']
        self.half_place_size = sim_config.get('half_place_size', 5)
        
        # Agent parameters
        agent_config = self.config['agents']
        self.num_agents = agent_config['num_agents']
        self.communication_radius = agent_config['communication_radius']
        self.memory_limit = agent_config.get('memory_limit', 20)
        self.memory_size = agent_config.get('memory_size', 5)
        self.message_history_limit = agent_config.get('message_history_limit', 10)
        self.message_context_size = agent_config.get('message_context_size', 3)
        
        # Place parameters - support multiple places
        if 'places' not in self.config:
            raise ValueError("No 'places' configuration found in config file. Please use 'places:' key.")
        
        self.places = self.config['places']
        
        # Validate places configuration
        if not isinstance(self.places, list):
            raise ValueError("'places' must be a list of place configurations.")
        
        if len(self.places) == 0:
            raise ValueError("At least one place must be configured in 'places'.")
        
        # Validate each place configuration
        required_fields = ['name', 'type', 'center_x', 'center_y', 'half_size', 'capacity']
        for i, place in enumerate(self.places):
            if not isinstance(place, dict):
                raise ValueError(f"Place at index {i} must be a dictionary.")
            
            for field in required_fields:
                if field not in place:
                    raise ValueError(f"Place at index {i} is missing required field: '{field}'")
        
        place_names = [place['name'] for place in self.places]
        place_types = [place['type'] for place in self.places]
        logger.info(f"Initialized {len(self.places)} place(s): {place_names} (types: {place_types})")
        
        # Fire parameters (multiple fires supported)
        fires_config = self.config.get('fires', [])
        self.fire_configs: List[Dict] = []
        for i, fc in enumerate(fires_config):
            config_entry = {
                'name': fc.get('name', f'fire_{i}'),
                'start_step': fc['start_step'],
                'intensity': fc['intensity'],
                'radius': fc['radius'],
            }
            if 'center_x' in fc and 'center_y' in fc:
                config_entry['center_x'] = fc['center_x']
                config_entry['center_y'] = fc['center_y']
            self.fire_configs.append(config_entry)
            pos_info = f"({fc['center_x']}, {fc['center_y']})" if 'center_x' in fc else "random"
            logger.info(
                f"Fire '{config_entry['name']}' configured: step={fc['start_step']}, "
                f"intensity={fc['intensity']}, radius={fc['radius']}, position={pos_info}"
            )
        self.fire_states: List[Dict] = []  # Active fires

        # Stock is per-run mutable state; the config value is only its start
        self.stock: Dict[str, Optional[int]] = {
            p['name']: p.get('stock') for p in self.places
        }
        self.purchases: List[Dict] = []  # {step, agent_id, place, price}

        # Ad campaigns: the information-axis counterpart of a fire. A fire is
        # perceived by distance; an ad is delivered by targeting.
        self.ad_configs: List[Dict] = self.config.get('ad_campaigns', [])
        self.impressions = 0  # Cumulative ad impressions (the thing you pay for)
        self.clicks = 0       # Cumulative click-throughs (the creative's first effect)
        self.searches = 0     # Cumulative organic arrivals (the channel an ad can cannibalise)
        self.mutes = 0        # Cumulative opt-outs (what over-exposure costs)
        self.ads_delivered_this_step: List[Dict] = []
        for ac in self.ad_configs:
            logger.info(
                f"Ad campaign '{ac.get('name')}' configured: "
                f"steps {ac['start_step']}-{ac.get('end_step', ac['start_step'])}, "
                f"targeting={ac.get('targeting', 'all')}"
            )

        # Budgets are handed out from the config in order so that the same
        # seed produces the same population in every scenario
        self.budgets: List[int] = agent_config.get('budgets', [])
        self.personas: List[str] = agent_config.get('personas', [])
        self.relations: List[List[int]] = agent_config.get('relations', [])
        self.active_from: List[int] = agent_config.get('active_from', [])

        # LLM parameters
        llm_config = self.config['llm']
        self.llm_client = OllamaClient(
            base_url=llm_config['base_url'],
            model=llm_config['model'],
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 200),
            repeat_penalty=llm_config.get('repeat_penalty', 1.1),
            repeat_last_n=llm_config.get('repeat_last_n', 128),
            min_p=llm_config.get('min_p', 0.05),
            seed=llm_config.get('seed'),
            # None when unset: the model keeps its own thinking default
            think=llm_config.get('think'),
            timeout_seconds=llm_config.get('timeout', 300)  # [s]
        )
        
        # Initialize agents
        self.agents: List[Agent] = []
        self.step = 0

    def _is_position_in_place(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside any place"""
        return get_place_at_position(position, self.places) is not None

    def _log_message(
        self,
        from_agent_id: int,
        to_agent_id: int,
        message: str,
        reasoning: str = ""
    ) -> None:
        """Log a message to messages.jsonl file"""
        if not self.output_dir:
            return

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        messages_file = os.path.join(self.output_dir, "messages.jsonl")
        record = {
            "step": self.step,
            "from": from_agent_id,
            "to": to_agent_id,
            "message": message,
            "reasoning": reasoning
        }

        with open(messages_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _log_memory_reasoning_batch(
        self,
        records: List[Dict]
    ) -> None:
        """Log memory and reasoning records in batch to memory_reasoning.jsonl file
        
        This is more efficient than writing one record at a time, especially
        when logging for all agents in each step.
        """
        if not self.output_dir or not records:
            return

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        memory_reasoning_file = os.path.join(self.output_dir, "memory_reasoning.jsonl")
        
        # Write all records at once (buffered I/O)
        with open(memory_reasoning_file, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _log_metrics(self) -> None:
        """Append one line of observable state per step to metrics.jsonl.

        Everything the analysis and the dashboard need is here, so neither has
        to re-read the LLM logs.
        """
        if not self.output_dir:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        record = {
            "step": self.step,
            "stock": dict(self.stock),
            "impressions_cum": self.impressions,
            "clicks_cum": self.clicks,
            "searches_cum": self.searches,
            "mutes_cum": self.mutes,
            "purchases_cum": len(self.purchases),
            "ads_delivered": self.ads_delivered_this_step,
            "agents": [
                {
                    "id": a.id,
                    "pos": list(a.position),
                    "place": a.current_place,
                    "budget": a.budget,
                    "ads_seen": len(a.ads_seen),
                    "clicks": a.clicks,
                    "searches": a.searches,
                    "muted": a.muted,
                    "intent": a.intent,
                    "heard_of": sorted(a.heard_places),
                    "in_market": self.step >= a.active_from,
                    "purchased_at": a.purchased_at,
                    "purchase_place": a.purchase_place,
                }
                for a in self.agents
            ],
        }
        with open(os.path.join(self.output_dir, "metrics.jsonl"), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _targeted_agents(self, ad: Dict) -> List[Agent]:
        """Who this campaign reaches this step.

        - "all": an untargeted broadcast, every agent
        - "in_market": agents already within `targeting_radius` of the sponsor's
          place. This is the simulator's stand-in for retargeting: spending the
          budget on people who are, by construction, closest to buying anyway
        """
        targeting = ad.get('targeting', 'all')

        if targeting == 'all':
            candidates = list(self.agents)
        elif targeting == 'in_market':
            place = next((p for p in self.places if p['name'] == ad['place']), None)
            if place is None:
                raise ValueError(f"Ad '{ad.get('name')}' targets unknown place '{ad.get('place')}'")
            center = (place['center_x'], place['center_y'])
            radius = ad.get('targeting_radius', 10)
            candidates = [a for a in self.agents if a.distance_to(center) <= radius]
        else:
            raise ValueError(f"Unknown targeting mode: {targeting}")

        # Nobody keeps paying to advertise a durable good to someone who has
        # already bought it, and a muted agent cannot be reached at any price.
        # The second exclusion is what gives over-exposure a real cost.
        return [a for a in candidates if a.purchased_at is None and not a.muted]

    def _deliver_ads(self) -> None:
        """Deliver every campaign active on this step and bill the impressions."""
        self.ads_delivered_this_step = []
        for ad in self.ad_configs:
            start = ad['start_step']
            end = ad.get('end_step', start)
            if not (start <= self.step <= end):
                continue

            recipients = self._targeted_agents(ad)
            for agent in recipients:
                agent.ads_seen.append({
                    'step': self.step,
                    'sponsor': ad.get('sponsor', ad.get('name', 'advertiser')),
                    'copy': ad['copy'],
                    'place': ad.get('place'),
                })
                if ad.get('place'):
                    agent.ad_places.add(ad['place'])
            self.impressions += len(recipients)
            self.ads_delivered_this_step.append({
                'name': ad.get('name'),
                'to': [a.id for a in recipients],
            })
            if recipients:
                logger.info(
                    f"Step {self.step}: ad '{ad.get('name')}' delivered to "
                    f"{len(recipients)} agent(s): {[a.id for a in recipients]}"
                )

    def _try_click(self, agent: Agent) -> bool:
        """Follow an ad: land inside the advertised place in one step.

        This is the ad's own causal channel. Walking the grid is what someone
        does who was already heading that way; clicking is what an ad buys.
        Keeping them separate is what lets the decomposition say whether a
        conversion came from the campaign or would have walked in regardless.
        """
        if agent.purchased_at is not None or not agent.ad_places:
            return False

        # An agent that saw ads for several places follows the most recent one
        target = next((ad.get('place') for ad in reversed(agent.ads_seen) if ad.get('place')), None)
        if not target:
            return False

        place = next((p for p in self.places if p['name'] == target), None)
        if place is None:
            return False

        agent.position = (place['center_x'], place['center_y'])
        agent.clicks += 1
        self.clicks += 1
        logger.info(
            f"Step {self.step}: Agent {agent.id} clicked through to {target} "
            f"(total clicks {self.clicks})"
        )
        return True

    def _try_search(self, agent: Agent) -> bool:
        """Go looking for a shop heard about through word of mouth.

        The organic counterpart of the click. An advertised world can convert
        the same person through the paid route instead, and the platform then
        books a conversion that would have arrived on its own - which is
        exactly what the decomposition is built to expose.
        """
        if agent.purchased_at is not None or not agent.heard_places:
            return False

        # Searching your way to the shop you are already standing in is a
        # no-op that an agent can repeat forever, so the current place is not
        # a candidate. Leaving it in produced agents that searched their way
        # into the same shop every step instead of deciding anything.
        known = [p for p in self.places
                 if p['name'] in agent.heard_places and p['name'] != agent.current_place]
        if not known:
            return False

        target = min(known, key=lambda p: agent.distance_to((p['center_x'], p['center_y'])))
        agent.position = (target['center_x'], target['center_y'])
        agent.searches += 1
        self.searches += 1
        logger.info(
            f"Step {self.step}: Agent {agent.id} searched its way to {target['name']} "
            f"(total organic arrivals {self.searches})"
        )
        return True

    def _try_mute(self, agent: Agent) -> bool:
        """Opt out of the advertiser. Frequency has a ceiling and this is it."""
        if agent.muted or not agent.ads_seen:
            return False
        agent.muted = True
        self.mutes += 1
        logger.info(
            f"Step {self.step}: Agent {agent.id} muted the advertiser after "
            f"{len(agent.ads_seen)} impressions (total mutes {self.mutes})"
        )
        return True

    def _try_purchase(self, agent: Agent) -> bool:
        """Execute a "buy" action if the world allows it.

        The item is durable, so one purchase per agent. That keeps a run
        comparable to its twin agent by agent: each agent either converted or
        did not.
        """
        if agent.purchased_at is not None or self.step < agent.active_from:
            return False
        if not agent.in_place or not agent.current_place:
            return False

        place = next((p for p in self.places if p['name'] == agent.current_place), None)
        if place is None:
            return False

        price = place.get('price')
        stock = self.stock.get(agent.current_place)
        if price is None or stock is None or stock <= 0 or agent.budget < price:
            return False

        self.stock[agent.current_place] = stock - 1
        agent.budget -= price
        agent.purchased_at = self.step
        agent.purchase_place = agent.current_place
        self.purchases.append({
            'step': self.step,
            'agent_id': agent.id,
            'place': agent.current_place,
            'price': price,
            'ads_seen_before_purchase': len(agent.ads_seen),
        })
        logger.info(
            f"Step {self.step}: Agent {agent.id} bought 1 unit at {agent.current_place} "
            f"(stock left {self.stock[agent.current_place]}, ads seen {len(agent.ads_seen)})"
        )
        return True

    def _generate_random_position(self) -> Tuple[int, int]:
        """Generate a random position within the space (origin-centered coordinate system)"""
        return (
            random.randint(-self.half_space_size, self.half_space_size),
            random.randint(-self.half_space_size, self.half_space_size)
        )
    
    def _generate_initial_positions(self, avoid_places: bool = True) -> List[Tuple[int, int]]:
        """Generate initial positions for agents"""
        positions: List[Tuple[int, int]] = []
        used_positions: Set[Tuple[int, int]] = set()
        attempts = 0
        
        while len(positions) < self.num_agents and attempts < MAX_POSITION_ATTEMPTS:
            position = self._generate_random_position()
            
            # Skip if position is already used
            if position in used_positions:
                attempts += 1
                continue
            
            # Skip if position is in any place and we want to avoid it
            if avoid_places and self._is_position_in_place(position):
                attempts += 1
                continue
            
            positions.append(position)
            used_positions.add(position)
            attempts += 1
        
        # If we couldn't generate enough positions avoiding places, fill remaining
        if len(positions) < self.num_agents:
            logger.warning(
                f"Could only generate {len(positions)} unique positions avoiding places. "
                "Using all available space."
            )
            while len(positions) < self.num_agents:
                position = self._generate_random_position()
                if position not in used_positions:
                    positions.append(position)
                    used_positions.add(position)
        
        return positions
    
    def initialize_agents(self):
        """Initialize agents at random positions"""
        logger.info(f"Initializing {self.num_agents} agents...")
        
        positions = self._generate_initial_positions(avoid_places=True)
        
        # Create agents
        for i in range(self.num_agents):
            gender = random.choice(["male", "female"])
            budget = self.budgets[i % len(self.budgets)] if self.budgets else 0
            persona = self.personas[i % len(self.personas)] if self.personas else ""
            relations = self.relations[i] if i < len(self.relations) else []
            active_from = self.active_from[i] if i < len(self.active_from) else 1
            agent = Agent(
                agent_id=i,
                initial_position=positions[i],
                llm_client=self.llm_client,
                communication_radius=self.communication_radius,
                half_space_size=self.half_space_size,
                places=self.places,
                num_agents=self.num_agents,
                gender=gender,
                budget=budget,
                persona=persona,
                relations=relations,
                active_from=active_from,
                memory_limit=self.memory_limit,
                memory_size=self.memory_size,
                message_history_limit=self.message_history_limit,
                message_context_size=self.message_context_size
            )
            agent.update_state()  # Initialize in_place state
            self.agents.append(agent)
        
        logger.info("Agents initialized successfully")
    
    def get_agents_in_place(self, place_name: Optional[str] = None) -> List[Agent]:
        """Get list of agents currently in a specific place or any place"""
        if place_name:
            return [agent for agent in self.agents if agent.current_place == place_name]
        return [agent for agent in self.agents if agent.in_place]
    
    def get_place_status(self, place_name: Optional[str] = None) -> Dict:
        """Get current place status for a specific place or overall status"""
        if place_name:
            # Get status for a specific place
            place_config = next((p for p in self.places if p['name'] == place_name), None)
            if not place_config:
                raise ValueError(f"Place '{place_name}' not found")
            
            agents_in_place = len(self.get_agents_in_place(place_name))
            capacity = place_config['capacity']
            occupancy_rate = agents_in_place / capacity

            return {
                "place_name": place_name,
                "agents_in_place": agents_in_place,
                "capacity": capacity,
                "occupancy_rate": occupancy_rate,
                "stock": self.stock.get(place_name),
                "price": place_config.get('price'),
            }
        else:
            # Get overall status (all places combined)
            agents_in_place = len(self.get_agents_in_place())
            occupancy_rate = agents_in_place / self.num_agents
            
            # Get per-place status (optimized: calculate directly instead of recursive calls)
            place_statuses = {}
            for place in self.places:
                place_agents = len(self.get_agents_in_place(place['name']))
                place_capacity = place['capacity']
                place_occupancy_rate = place_agents / place_capacity

                place_statuses[place['name']] = {
                    "place_name": place['name'],
                    "agents_in_place": place_agents,
                    "capacity": place_capacity,
                    "occupancy_rate": place_occupancy_rate,
                    "stock": self.stock.get(place['name']),
                    "price": place.get('price'),
                }
            
            return {
                "agents_in_place": agents_in_place,
                "occupancy_rate": occupancy_rate,
                "places": place_statuses
            }
    
    def get_fire_info_for_agent(self, agent: Agent) -> Optional[List[Dict]]:
        """Return list of perceived fire info dicts, or None if no fires perceived.

        Implements Model B: only agents within each fire's radius get that fire's data.
        Agents outside all radii must learn about fires through messages.
        """
        if not self.fire_states:
            return None

        perceived = []
        for fire in self.fire_states:
            if not fire.get('active'):
                continue
            fire_pos = fire['position']
            distance = agent.distance_to(fire_pos)
            if distance <= fire['radius']:
                perceived.append({
                    'name': fire['name'],
                    'fire_position': fire_pos,
                    'intensity': fire['intensity'],
                    'radius': fire['radius'],
                    'agent_distance': round(distance, 2),
                })
        return perceived if perceived else None

    def step_simulation(self):
        """Execute one simulation step

        New order:
        1. All agents decide messages (without position information)
        2. Messages are sent to nearby agents (using decision-time positions)
        3. All agents decide actions (with position information and message content)
        4. Agents move to new positions
        """
        self.step += 1

        # Fire activation check (multiple fires)
        active_names = {f['name'] for f in self.fire_states}
        for fc in self.fire_configs:
            if fc['name'] not in active_names and self.step >= fc['start_step']:
                if 'center_x' in fc and 'center_y' in fc:
                    fire_pos = (fc['center_x'], fc['center_y'])
                else:
                    fire_pos = self._generate_random_position()
                fire_state = {
                    'name': fc['name'],
                    'position': fire_pos,
                    'intensity': fc['intensity'],
                    'radius': fc['radius'],
                    'start_step': fc['start_step'],
                    'active': True,
                }
                self.fire_states.append(fire_state)
                logger.info(
                    f"FIRE '{fc['name']}' started at position {fire_pos} with intensity "
                    f"{fc['intensity']}, radius {fc['radius']}"
                )

        # Ad delivery happens before any reasoning, so the copy is already in
        # the prompt for both the message phase and the action phase
        self._deliver_ads()

        # Update agent states
        for agent in self.agents:
            agent.update_state(self.places)

        # Phase 1: Collect message decisions from all agents (without position information)
        message_decisions = []
        for agent in self.agents:
            nearby_agents = agent.get_nearby_agents(self.agents)
            # Get place status for the place the agent is in (or None if outside)
            agent_place_status = None
            if agent.in_place and agent.current_place:
                agent_place_status = self.get_place_status(agent.current_place)
            fire_info = self.get_fire_info_for_agent(agent)
            message_decision = agent.decide_message(agent_place_status, nearby_agents, self.step, fire_info=fire_info)
            message_decisions.append((agent, message_decision, nearby_agents))

        # Phase 2: Send messages (using decision-time nearby agents, before movement)
        for agent, message_decision, nearby_agents in message_decisions:
            message_content = message_decision.get('message', '')
            if message_content and nearby_agents:
                logger.info(
                    f"Step {self.step}: Agent {agent.id} sends message to {len(nearby_agents)} nearby agent(s): "
                    f"\"{message_content}\""
                )
                for other_agent in nearby_agents:
                    other_agent.receive_message(agent.id, message_content, step=self.step)
                    other_agent.note_places_mentioned(message_content)
                    # Log message to jsonl file
                    self._log_message(
                        from_agent_id=agent.id,
                        to_agent_id=other_agent.id,
                        message=message_content,
                        reasoning=message_decision.get('reasoning', '')
                    )

        # Phase 3: Collect action decisions from all agents (with position information and message content)
        action_decisions = []
        memory_reasoning_records = []  # Batch records for efficient I/O
        for agent, message_decision, nearby_agents in message_decisions:
            # Get place status for the place the agent is in (or None if outside)
            agent_place_status = None
            if agent.in_place and agent.current_place:
                agent_place_status = self.get_place_status(agent.current_place)
            message_content = message_decision.get('message', '')
            fire_info = self.get_fire_info_for_agent(agent)
            action_decision = agent.decide_action(agent_place_status, nearby_agents, self.step, message_content, fire_info=fire_info)
            action_decisions.append((agent, action_decision))
            
            # Collect memory and reasoning records for batch writing
            memory_reasoning_records.append({
                "step": self.step,
                "id": agent.id,
                "memory": action_decision.get('memory', ''),
                "reasoning": action_decision.get('reasoning', '')
            })
        
        # Write all memory/reasoning records in batch (more efficient than individual writes)
        self._log_memory_reasoning_batch(memory_reasoning_records)

        # Phase 4: Execute movement and purchases
        for agent, action_decision in action_decisions:
            action = action_decision['action']
            if action_decision.get('intent') is not None:
                agent.intent = action_decision['intent']

            if action == 'buy':
                self._try_purchase(agent)
            elif action == 'click':
                self._try_click(agent)
            elif action == 'search':
                self._try_search(agent)
            elif action == 'mute':
                self._try_mute(agent)
            elif action == 'move' and action_decision['direction']:
                agent.move(action_decision['direction'])

        # Update states after movement
        for agent in self.agents:
            agent.update_state(self.places)

        self._log_metrics()

        if self.step % LOG_INTERVAL == 0:
            # The overall status carries both the total and the per-place counts,
            # so one call covers the whole line.
            overall_status = self.get_place_status()
            place_info = ", ".join([
                f"{place_name}: {status['agents_in_place']}"
                for place_name, status in overall_status['places'].items()
            ])
            logger.info(
                f"Step {self.step}/{self.duration}: "
                f"{overall_status['agents_in_place']} agents in places ({place_info}), "
                f"{overall_status['occupancy_rate']:.1%} overall occupancy"
            )


