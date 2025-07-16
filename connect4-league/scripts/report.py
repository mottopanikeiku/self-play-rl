#!/usr/bin/env python3
"""Report Generator

Creates HTML reports with visualizations and GIFs from tournament results.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import io
import base64

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from env.connect4 import Connect4Env


def load_tournament_data() -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Load tournament results and Elo history."""
    data_dir = Path("data")
    
    # Load results
    results_file = data_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load Elo history
    elo_file = data_dir / "elo.csv"
    elo_history = {}
    
    if elo_file.exists():
        with open(elo_file, 'r') as f:
            reader = csv.DictReader(f)
            
            # Initialize agent lists
            agent_names = [col for col in reader.fieldnames if col != 'game']
            for agent in agent_names:
                elo_history[agent] = []
            
            games = []
            for row in reader:
                games.append(int(row['game']))
                for agent in agent_names:
                    elo_history[agent].append(float(row[agent]))
            
            elo_history['games'] = games
    
    return results, elo_history


def create_winrate_heatmap(results: List[Dict[str, Any]]) -> str:
    """Create win rate heatmap and return as base64 encoded image."""
    # Get all agents
    agents = set()
    for result in results:
        agents.add(result['agent1'])
        agents.add(result['agent2'])
    
    agents = sorted(list(agents))
    n_agents = len(agents)
    
    # Create win rate matrix
    win_matrix = np.zeros((n_agents, n_agents))
    game_counts = np.zeros((n_agents, n_agents))
    
    for result in results:
        i = agents.index(result['agent1'])
        j = agents.index(result['agent2'])
        
        total_games = result['wins'] + result['losses'] + result['draws']
        if total_games > 0:
            win_rate = result['wins'] / total_games
            win_matrix[i, j] = win_rate
            game_counts[i, j] = total_games
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Mask diagonal (agents don't play themselves)
    mask = np.eye(n_agents, dtype=bool)
    
    sns.heatmap(
        win_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0.5,
        square=True,
        xticklabels=agents,
        yticklabels=agents,
        cbar_kws={'label': 'Win Rate'}
    )
    
    plt.title('Agent Win Rates (Row vs Column)', fontsize=16, fontweight='bold')
    plt.xlabel('Opponent', fontsize=12)
    plt.ylabel('Agent', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def create_elo_evolution_plot(elo_history: Dict[str, List[float]]) -> str:
    """Create Elo evolution plot and return as base64 encoded image."""
    if not elo_history or 'games' not in elo_history:
        # Create empty plot
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, 'No Elo history available', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16)
        plt.title('Elo Rating Evolution')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
    
    plt.figure(figsize=(12, 6))
    
    games = elo_history['games']
    colors = plt.cm.tab10(np.linspace(0, 1, len(elo_history) - 1))
    
    for i, (agent, ratings) in enumerate(elo_history.items()):
        if agent != 'games':
            plt.plot(games, ratings, label=agent, color=colors[i % len(colors)], 
                    linewidth=2, marker='o', markersize=3, alpha=0.8)
    
    plt.xlabel('Games Played', fontsize=12)
    plt.ylabel('Elo Rating', fontsize=12)
    plt.title('Elo Rating Evolution Throughout Tournament', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


def create_sample_game_gif() -> str:
    """Create a sample Connect-4 game GIF."""
    # Create a simple demo game
    env = Connect4Env()
    env.reset()
    
    frames = []
    
    # Sample moves for a quick game
    moves = [3, 3, 2, 4, 1, 5, 0]  # Some reasonable moves
    
    # Add initial empty board
    frames.append(render_board_to_image(env.board))
    
    for i, move in enumerate(moves):
        if move in env.legal_actions():
            env.step(move)
            frames.append(render_board_to_image(env.board))
            
            if env.game_over:
                break
    
    # Create GIF
    gif_path = Path("data") / "sample_game.gif"
    gif_path.parent.mkdir(exist_ok=True)
    
    if frames:
        imageio.mimsave(gif_path, frames, duration=0.8, loop=0)
        return str(gif_path.name)
    
    return "No game GIF available"


def render_board_to_image(board: np.ndarray) -> np.ndarray:
    """Render Connect-4 board to image array."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Create color map: 0=white, 1=red, 2=yellow
    colors = np.zeros((*board.shape, 3))
    colors[board == 1] = [1, 0, 0]  # Red for player 1
    colors[board == 2] = [1, 1, 0]  # Yellow for player 2
    colors[board == 0] = [0.9, 0.9, 0.9]  # Light gray for empty
    
    ax.imshow(colors, aspect='equal')
    
    # Add grid
    for i in range(board.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=2)
    for j in range(board.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=2)
    
    # Add column numbers
    for j in range(board.shape[1]):
        ax.text(j, -0.7, str(j), ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.set_xlim(-0.5, board.shape[1] - 0.5)
    ax.set_ylim(board.shape[0] - 0.5, -1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Connect-4 Board', fontsize=16, fontweight='bold')
    
    # Convert to image array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image


def generate_html_report(
    results: List[Dict[str, Any]], 
    elo_history: Dict[str, List[float]],
    heatmap_b64: str,
    elo_plot_b64: str,
    gif_filename: str
) -> str:
    """Generate HTML report."""
    
    # Calculate summary statistics
    total_games = sum(r['wins'] + r['losses'] + r['draws'] for r in results)
    total_matchups = len(results)
    
    # Get final Elo ratings
    final_ratings = {}
    if elo_history and 'games' in elo_history:
        for agent, ratings in elo_history.items():
            if agent != 'games' and ratings:
                final_ratings[agent] = ratings[-1]
    
    # Sort agents by final Elo
    sorted_agents = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Connect-4 Tournament Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2980b9;
        }}
        .leaderboard {{
            background-color: #fff;
            border-radius: 8px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .leaderboard table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .leaderboard th {{
            background-color: #3498db;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        .leaderboard td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .leaderboard tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .rank-1 {{ background-color: #f1c40f !important; font-weight: bold; }}
        .rank-2 {{ background-color: #e8e8e8 !important; font-weight: bold; }}
        .rank-3 {{ background-color: #cd7f32 !important; font-weight: bold; }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .gif-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .matchup-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .matchup-card {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .vs-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            text-align: center;
        }}
        .score {{
            font-size: 1.2em;
            text-align: center;
            margin: 5px 0;
        }}
        .wins {{ color: #27ae60; }}
        .losses {{ color: #e74c3c; }}
        .draws {{ color: #f39c12; }}
        footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔴🟡 Connect-4 Tournament Report</h1>
        
        <div class="summary">
            <h2>Tournament Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-value">{len(sorted_agents)}</div>
                    <div>Participants</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{total_matchups}</div>
                    <div>Matchups</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{total_games}</div>
                    <div>Total Games</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">{total_games/total_matchups if total_matchups > 0 else 0:.0f}</div>
                    <div>Games per Matchup</div>
                </div>
            </div>
        </div>

        <div class="leaderboard">
            <h2>🏆 Final Leaderboard</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Agent</th>
                        <th>Elo Rating</th>
                        <th>Rating Change</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for rank, (agent, rating) in enumerate(sorted_agents, 1):
        rating_change = rating - 1200  # Starting rating was 1200
        change_color = "green" if rating_change > 0 else "red" if rating_change < 0 else "gray"
        change_sign = "+" if rating_change > 0 else ""
        
        row_class = ""
        if rank == 1:
            row_class = "rank-1"
        elif rank == 2:
            row_class = "rank-2"
        elif rank == 3:
            row_class = "rank-3"
        
        html += f"""
                    <tr class="{row_class}">
                        <td>{rank}</td>
                        <td>{agent}</td>
                        <td>{rating:.1f}</td>
                        <td style="color: {change_color};">{change_sign}{rating_change:.1f}</td>
                    </tr>
        """
    
    html += f"""
                </tbody>
            </table>
        </div>

        <div class="chart-container">
            <h2>📊 Win Rate Heat Map</h2>
            <p>Win rates for each agent (row) against each opponent (column)</p>
            <img src="data:image/png;base64,{heatmap_b64}" alt="Win Rate Heatmap">
        </div>

        <div class="chart-container">
            <h2>📈 Elo Rating Evolution</h2>
            <p>How agent ratings changed throughout the tournament</p>
            <img src="data:image/png;base64,{elo_plot_b64}" alt="Elo Evolution Plot">
        </div>

        <div class="gif-container">
            <h2>🎮 Sample Game</h2>
            <p>Example Connect-4 gameplay</p>
            <p><em>GIF: {gif_filename}</em></p>
        </div>

        <div class="matchup-grid">
            <h2>⚔️ Head-to-Head Results</h2>
    """
    
    for result in results:
        wins = result['wins']
        losses = result['losses']
        draws = result['draws']
        total = wins + losses + draws
        win_rate = wins / total if total > 0 else 0
        
        html += f"""
            <div class="matchup-card">
                <div class="vs-header">{result['agent1']} vs {result['agent2']}</div>
                <div class="score wins">Wins: {wins}</div>
                <div class="score losses">Losses: {losses}</div>
                <div class="score draws">Draws: {draws}</div>
                <div class="score">Win Rate: {win_rate:.3f}</div>
            </div>
        """
    
    html += f"""
        </div>

        <footer>
            <p>Generated by Connect-4 Tournament System</p>
            <p>Report created on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
    """
    
    return html


def main() -> None:
    """Main report generation function."""
    print("Generating Connect-4 Tournament Report...")
    
    try:
        # Load data
        print("Loading tournament data...")
        results, elo_history = load_tournament_data()
        
        # Create visualizations
        print("Creating win rate heatmap...")
        heatmap_b64 = create_winrate_heatmap(results)
        
        print("Creating Elo evolution plot...")
        elo_plot_b64 = create_elo_evolution_plot(elo_history)
        
        print("Creating sample game GIF...")
        gif_filename = create_sample_game_gif()
        
        # Generate HTML report
        print("Generating HTML report...")
        html_content = generate_html_report(
            results, elo_history, heatmap_b64, elo_plot_b64, gif_filename
        )
        
        # Save report
        report_dir = Path("report")
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / "index.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Report generated successfully!")
        print(f"📄 Report saved to: {report_file.absolute()}")
        print(f"🌐 Open in browser: file://{report_file.absolute()}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 