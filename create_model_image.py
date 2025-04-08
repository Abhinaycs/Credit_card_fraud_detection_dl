import matplotlib.pyplot as plt
import numpy as np

def create_model_architecture_image():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define layer sizes
    layers = [30, 128, 64, 32, 16, 8, 4, 2, 1]  # Input + 7 hidden + output
    layer_names = ['Input\n(30)', 'Hidden\n(128)', 'Hidden\n(64)', 'Hidden\n(32)', 
                  'Hidden\n(16)', 'Hidden\n(8)', 'Hidden\n(4)', 'Hidden\n(2)', 'Output\n(1)']
    
    # Calculate positions
    x_positions = np.linspace(0, 1, len(layers))
    y_positions = np.linspace(0, 1, max(layers))
    
    # Plot nodes
    for i, (layer_size, x) in enumerate(zip(layers, x_positions)):
        # Calculate y positions for this layer
        y_layer = np.linspace(0.1, 0.9, layer_size)
        
        # Plot nodes
        ax.scatter([x] * layer_size, y_layer, s=100, c='lightblue', edgecolor='black')
        
        # Add layer name
        ax.text(x, 0.95, layer_names[i], ha='center', va='top', fontweight='bold')
    
    # Plot connections
    for i in range(len(layers)-1):
        x1 = x_positions[i]
        x2 = x_positions[i+1]
        y1 = np.linspace(0.1, 0.9, layers[i])
        y2 = np.linspace(0.1, 0.9, layers[i+1])
        
        for y_1 in y1:
            for y_2 in y2:
                ax.plot([x1, x2], [y_1, y_2], 'gray', alpha=0.1)
    
    # Customize plot
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('BiLSTM Model Architecture', pad=20, fontsize=14)
    
    # Add annotations
    plt.text(0.5, -0.1, 'LeakyReLU Activation • Batch Normalization • Dropout', 
             ha='center', va='center', fontsize=10)
    
    # Save the figure
    plt.savefig('model_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    create_model_architecture_image() 