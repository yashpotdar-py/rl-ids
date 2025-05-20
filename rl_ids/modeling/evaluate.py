import torch                                       # :contentReference[oaicite:15]{index=15}
import pandas as pd
from sklearn.metrics import classification_report, f1_score  # :contentReference[oaicite:16]{index=16}
from rl_ids.environment import IntrusionEnv
from rl_ids.modeling.dqn_agent import DQNAgent

def load_agent(checkpoint_path: str, state_dim, action_dim):
    # Load the checkpoint first to check its architecture
    ckpt = torch.load(checkpoint_path, weights_only=False)
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    
    # Extract hidden layer dimensions from the hyperparams if available
    hidden_dims = [128, 128]  # Default to a deeper network architecture
    if 'hyperparams' in ckpt and 'hidden_dims' in ckpt['hyperparams']:
        hidden_dims = ckpt['hyperparams']['hidden_dims']
    
    # Create agent with matching architecture
    agent = DQNAgent(state_dim, action_dim, hidden_dims=hidden_dims)
    
    # Load the state dict
    if 'q_net_state_dict' in ckpt:
        agent.q_net.load_state_dict(ckpt['q_net_state_dict'])
    elif 'model_state_dict' in ckpt:
        agent.q_net.load_state_dict(ckpt['model_state_dict'])
    elif 'policy_net_state_dict' in ckpt:
        agent.q_net.load_state_dict(ckpt['policy_net_state_dict'])
    else:
        raise KeyError(f"Could not find model weights in checkpoint")
    
    agent.q_net.eval()
    return agent

def evaluate(agent, data_path: str):
    env = IntrusionEnv(data_path)
    y_true, y_pred = [], []
    state, _ = env.reset()
    done = False
    
    # Get the device the model is on
    device = next(agent.q_net.parameters()).device
    print(f"Model is on device: {device}")
    
    while not done:
        with torch.no_grad():
            # Move input tensor to the same device as the model
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
            action = int(agent.q_net(state_tensor).argmax())
        y_pred.append(action)
        # Fixed to use protected attributes
        y_true.append(env._y[env._idx])
        state, _, done, _, _ = env.step(action)
    print(classification_report(y_true, y_pred, target_names=['Benign','Attack']))
    print("F1 Score:", f1_score(y_true, y_pred))

if __name__ == "__main__":
    ckpt = "models/dqn/dqn/final_model.tar"
    data = "data/processed/cleaned.parquet"
    df = pd.read_parquet(data)                       # quick dimension check :contentReference[oaicite:17]{index=17}
    agent = load_agent(ckpt, df.shape[1]-1, 2)
    evaluate(agent, data)
