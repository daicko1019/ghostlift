"""
Visualization for LLM Multi-Agent 2D Simulation

Rendering is file-only: every figure is written to a PNG under the output
directory and no window is ever opened. The non-GUI Agg backend is therefore
selected unconditionally - it is always available and needs no display, so a
run behaves identically on macOS, Linux, WSL, an SSH session and CI.
"""
import matplotlib
import logging
from typing import List, Dict, Tuple, Optional

# Must be selected before pyplot is imported below.
matplotlib.use('Agg')

# Visualization constants
FIGURE_SIZE = (10, 10)
DPI = 150

# Agent visualization constants
AGENT_SIZE_IN_PLACE = 100
AGENT_SIZE_OUTSIDE = 80
AGENT_ALPHA = 0.7
COMMUNICATION_LINK_ALPHA = 0.3

# Place visualization constants
PLACE_LINEWIDTH = 2
PLACE_ALPHA = 0.3

# Fire visualization constants
FIRE_MARKER_SIZE = 200
FIRE_CIRCLE_ALPHA = 0.15
FIRE_CIRCLE_LINEWIDTH = 2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization class for simulation"""

    def __init__(self, half_space_size: int, places: List[Dict], num_agents: int = None):
        self.half_space_size = half_space_size
        self.places = places
        self.num_agents = num_agents
        self.fig = None
        self.ax = None

    def setup_figure(self):
        """Create a fresh figure for one frame"""
        # Every frame gets its own figure and is closed after being saved, so
        # a long run does not accumulate open figures.
        self.fig, self.ax = plt.subplots(figsize=FIGURE_SIZE)

        # Set up axes properties (origin-centered coordinate system)
        self.ax.set_xlim(-self.half_space_size, self.half_space_size)
        self.ax.set_ylim(-self.half_space_size, self.half_space_size)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, alpha=0.3)

    def draw_places(self):
        """Draw all place areas (cafes, libraries, etc.)"""
        # Color palette for different place types
        place_type_colors = {
            'cafe': 'lightcoral',
            'library': 'lightgreen',
            'restaurant': 'lightyellow',
            'park': 'lightpink'
        }
        # Fallback colors for unknown types
        default_colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
        
        for i, place in enumerate(self.places):
            half_size = place['half_size']
            center_x = place['center_x']
            center_y = place['center_y']
            if 'name' not in place:
                raise ValueError(f"Place at index {i} is missing required field: 'name'")
            place_name = place['name']
            place_type = place['type']
            
            # Choose color based on place type
            if place_type in place_type_colors:
                face_color = place_type_colors[place_type]
            else:
                face_color = default_colors[i % len(default_colors)]
            
            # Place covers -half_size to +half_size from center (inclusive)
            # Rectangle width/height = 2 * half_size + 1 to cover all cells
            place_width = 2 * half_size + 1
            place_rect = patches.Rectangle(
                (center_x - half_size - 0.5, center_y - half_size - 0.5),  # Bottom-left corner
                place_width,
                place_width,
                linewidth=PLACE_LINEWIDTH,
                edgecolor='blue',
                facecolor=face_color,
                alpha=PLACE_ALPHA,
                label=f"{place_name} ({place_type})"
            )
            self.ax.add_patch(place_rect)
            
            # Add place name and type label at center
            label_text = f"{place_name}\n({place_type})"
            self.ax.text(
                center_x,
                center_y,
                label_text,
                fontsize=9,
                ha='center',
                va='center',
                weight='bold',
                color='darkblue'
            )
    
    def draw_fires(self, fire_states: List[Dict]):
        """Draw fire center markers and perception radius circles for all active fires.

        Fire areas are colored using a colormap based on intensity (0.0-1.0).
        """
        fire_cmap = matplotlib.colormaps['YlOrRd']

        for fire in fire_states:
            if not fire.get('active'):
                continue

            fx, fy = fire['position']
            radius = fire['radius']
            intensity = fire['intensity']
            name = fire.get('name', 'fire')

            # Map intensity to color via colormap
            face_color = fire_cmap(intensity)

            # Draw perception radius circle with intensity-based color
            fire_circle = patches.Circle(
                (fx, fy),
                radius,
                linewidth=FIRE_CIRCLE_LINEWIDTH,
                edgecolor='red',
                facecolor=face_color,
                alpha=FIRE_CIRCLE_ALPHA + 0.1,
                linestyle='--',
            )
            self.ax.add_patch(fire_circle)

            # Draw fire center marker
            self.ax.scatter(
                fx, fy,
                c='red',
                s=FIRE_MARKER_SIZE,
                marker='^',
                edgecolors='darkred',
                linewidths=2,
                zorder=10,
            )

            # Label
            self.ax.text(
                fx, fy - 1.5,
                f'{name}\n(int={intensity})',
                fontsize=8,
                ha='center',
                va='top',
                color='darkred',
                fontweight='bold',
            )

    def draw_agents(
        self,
        agents: List,
        agents_by_place: Dict[str, List[int]],
        communication_links: List[Tuple[int, int]] = None
    ):
        """Draw agents and communication links"""
        # Draw communication links
        if communication_links:
            for agent_id1, agent_id2 in communication_links:
                agent1 = agents[agent_id1]
                agent2 = agents[agent_id2]
                self.ax.plot(
                    [agent1.position[0], agent2.position[0]],
                    [agent1.position[1], agent2.position[1]],
                    'gray',
                    alpha=COMMUNICATION_LINK_ALPHA,
                    linewidth=1
                )
        
        # Draw agents: color by gender (male=blue, female=red), marker by location (in place=★, outside=●)
        for agent in agents:
            color = 'blue' if agent.gender == 'male' else 'red'
            if agent.in_place and agent.current_place:
                marker = '*'  # Star for agents in a place
                size = AGENT_SIZE_IN_PLACE * 1.5  # Stars need larger size to be visible
            else:
                marker = 'o'  # Circle for agents outside places
                size = AGENT_SIZE_OUTSIDE

            self.ax.scatter(
                agent.position[0],
                agent.position[1],
                c=color,
                s=size,
                marker=marker,
                alpha=AGENT_ALPHA,
                edgecolors='black',
                linewidths=1
            )
            
            # Add agent ID label
            self.ax.text(
                agent.position[0] + 0.5,
                agent.position[1] + 0.5,
                str(agent.id),
                fontsize=8,
                ha='left'
            )
    
    def _group_agent_ids_by_place(self, agents: List) -> Dict[str, List[int]]:
        """Map each place name to the ids of the agents currently inside it"""
        return {
            place['name']: [
                agent.id for agent in agents
                if agent.in_place and agent.current_place == place['name']
            ]
            for place in self.places
        }

    def _find_communication_links(
        self,
        agents: List,
        communication_radius: Optional[float]
    ) -> List[Tuple[int, int]]:
        """Find agent pairs that can talk: within radius AND in the same area"""
        if not communication_radius:
            return []

        links = []
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i + 1:]:
                # Same area means both outside places, or both in the same one.
                # Agents in different places cannot hear each other even if
                # their coordinates happen to be close.
                same_area = (
                    (not agent1.in_place and not agent2.in_place) or
                    (agent1.in_place and agent2.in_place and
                     agent1.current_place == agent2.current_place)
                )
                if same_area and agent1.distance_to(agent2.position) <= communication_radius:
                    links.append((agent1.id, agent2.id))
        return links

    def _format_occupancy(self, place_status: Dict) -> str:
        """Format the occupancy part of the frame title"""
        if 'places' not in place_status:
            # Single place format
            return (
                f"Agents in place: {place_status['agents_in_place']}"
                f"/{place_status['capacity']} "
                f"({place_status['occupancy_rate']:.1%})"
            )

        # Multiple places format: overall count first, then a per-place summary
        place_info = [
            f"{place_name}: {status['agents_in_place']}/{status['capacity']} "
            f"({status['occupancy_rate']:.0%})"
            for place_name, status in place_status['places'].items()
        ]
        return (
            f"Total in places: {place_status['agents_in_place']} "
            f"({place_status['occupancy_rate']:.1%}) | "
            f"{' | '.join(place_info)}"
        )

    def _count_agents_in_fire(self, agents: List, active_fires: List[Dict]) -> int:
        """Count agents inside at least one fire radius

        Counted over a set of ids: an agent standing inside several overlapping
        fire radii is still one exposed agent.
        """
        return len({
            agent.id for fire in active_fires for agent in agents
            if agent.distance_to(fire['position']) <= fire['radius']
        })

    def _build_title(
        self,
        place_status: Dict,
        step: int,
        agents: List,
        active_fires: List[Dict]
    ) -> str:
        """Build the frame title from occupancy and fire exposure counts"""
        title = f"Step {step} | {self._format_occupancy(place_status)}"

        if active_fires:
            title += f" | Fire: {self._count_agents_in_fire(agents, active_fires)} in radius"

        return title

    def _add_legend(self, active_fires: List[Dict]):
        """Add the legend: gender by color, location by marker, plus fires"""
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Male (outside)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Female (outside)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=12, label='Male (in place)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12, label='Female (in place)'),
        ]
        for fire in active_fires:
            legend_elements.append(
                Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                       markeredgecolor='darkred', markersize=10,
                       label=f"{fire.get('name', 'Fire')} (int={fire['intensity']})")
            )
        if active_fires:
            legend_elements.append(
                Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                       alpha=0.5, label='Perception radius')
            )

        # Add place area legends registered by draw_places
        legend_elements.extend(self.ax.get_legend_handles_labels()[0])
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def _add_fire_colorbar(self):
        """Add the fire intensity colorbar (fixed 0-1 range, shown from step 0)"""
        # make_axes_locatable matches the colorbar height to the plot's y-axis
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=matplotlib.colormaps['YlOrRd'],
            norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
        )
        scalar_mappable.set_array([])
        divider = make_axes_locatable(self.ax)
        colorbar_axes = divider.append_axes("right", size="3%", pad=0.1)
        colorbar = self.fig.colorbar(scalar_mappable, cax=colorbar_axes)
        colorbar.set_label('Intensity of Fire', fontsize=10)

    def visualize_step(
        self,
        agents: List,
        place_status: Dict,
        step: int,
        save_path: str,
        communication_radius: float = None,
        fire_states: Optional[List[Dict]] = None
    ):
        """Draw one simulation step and write it to save_path as a PNG"""
        active_fires = [fire for fire in (fire_states or []) if fire.get('active')]

        self.setup_figure()
        self.draw_places()
        self.draw_fires(fire_states or [])
        self.draw_agents(
            agents,
            self._group_agent_ids_by_place(agents),
            self._find_communication_links(agents, communication_radius)
        )
        title = self._build_title(place_status, step, agents, active_fires)
        self.ax.set_title(title, fontsize=11, fontweight='bold')
        self._add_legend(active_fires)
        self._add_fire_colorbar()

        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')

        # Release the figure now that it is on disk; nothing reuses it.
        plt.close(self.fig)
        self.fig = None
        self.ax = None
