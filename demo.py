import torch
import numpy as np
from pettingzoo.classic import connect_four_v3
from connect4_selfplay import Net

def load_model(checkpoint_path):
    model = Net()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def play_against_model(model):
    env = connect_four_v3.env(render_mode="human")
    env.reset()
    
    print("Starting game! You are player_0, model is player_1")
    print("Valid actions are columns 0-6")
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        
        if term or trunc:
            action = None
        else:
            if agent == 'player_0':
                print(f"\nCurrent board:")
                print(obs['observation'].reshape(6, 7))
                print(f"Valid moves: {[i for i, valid in enumerate(obs['action_mask']) if valid]}")
                
                while True:
                    try:
                        action = int(input("Enter your move (0-6): "))
                        if obs['action_mask'][action] == 1:
                            break
                        else:
                            print("Invalid move! Try again.")
                    except (ValueError, IndexError):
                        print("Please enter a valid column number (0-6)")
            
            else:
                action_mask = torch.tensor(obs['action_mask']).unsqueeze(0)
                state = torch.tensor(obs['observation']).unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(state, action_mask)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                
                print(f"Model plays column {action}")
        
        env.step(action)
    
    final_rewards = env.rewards
    if final_rewards['player_0'] == 1:
        print("\nYou win!")
    elif final_rewards['player_0'] == -1:
        print("\nModel wins!")
    else:
        print("\nIt's a draw!")

def watch_model_vs_model(model1, model2=None):
    if model2 is None:
        model2 = model1
    
    env = connect_four_v3.env(render_mode="human")
    env.reset()
    
    print("Watching model vs model...")
    input("Press Enter to start...")
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        
        if term or trunc:
            action = None
        else:
            current_model = model1 if agent == 'player_0' else model2
            
            action_mask = torch.tensor(obs['action_mask']).unsqueeze(0)
            state = torch.tensor(obs['observation']).unsqueeze(0)
            
            with torch.no_grad():
                logits = current_model(state, action_mask)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            print(f"{agent} plays column {action}")
            input("Press Enter for next move...")
        
        env.step(action)
    
    final_rewards = env.rewards
    if final_rewards['player_0'] == 1:
        print("\nPlayer 0 (Model 1) wins!")
    elif final_rewards['player_0'] == -1:
        print("\nPlayer 1 (Model 2) wins!")
    else:
        print("\nIt's a draw!")

def main():
    print("Connect-4 Demo")
    print("=" * 30)
    
    checkpoint_path = None
    for epoch in range(900, -1, -100):
        try:
            checkpoint_path = f'checkpoints/ckpt_{epoch}.pt'
            model = load_model(checkpoint_path)
            print(f"Loaded model from {checkpoint_path}")
            break
        except FileNotFoundError:
            continue
    
    if checkpoint_path is None:
        print("No trained model found! Please run connect4_selfplay.py first.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Play against the model")
        print("2. Watch model vs model")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            play_against_model(model)
        elif choice == '2':
            watch_model_vs_model(model)
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 