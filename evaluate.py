import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from retriever.search import FashionRetriever

def plot_results(query: str, results: list, save_path: str):
    # Ensure there are at least 1 result, else create empty plot
    if not results:
        print(f"No results found for query: {query}")
        return
        
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results, figsize=(4 * num_results, 5))
    
    # If there's only 1 result, axes is not a list
    if num_results == 1:
        axes = [axes]
        
    fig.suptitle(f"Query: {query}", fontsize=16)
    
    for ax, res in zip(axes, results):
        try:
            img = Image.open(res['path'])
            ax.imshow(img)
            score_text = f"Sim: {res['vector_sim']:.2f}\nAttr: {res['attr_score']:.2f}\nFinal: {res['final_score']:.2f}"
            ax.set_title(score_text, fontsize=10)
        except Exception as e:
            print(f"Error loading image {res['path']}: {e}")
            
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved result grid to {save_path}")

def run_evaluation():
    queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting."
    ]
    
    retriever = FashionRetriever()
    
    for i, query in enumerate(queries):
        print(f"\n--- Evaluating Query {i+1} ---")
        results = retriever.search(query, top_k=5)
        
        save_filename = f"query_{i+1}.png"
        save_path = os.path.join(config.RESULTS_DIR, save_filename)
        
        plot_results(query, results, save_path)

if __name__ == "__main__":
    run_evaluation()
