#!/usr/bin/env python3
"""
Generate Figure 3: LAALM Evaluation Pipeline Flowchart
Shows the modular inference workflow with confidence-aware semantic alignment
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def create_figure3(output_path="paper_figure3.png"):
    """Create the LAALM evaluation pipeline diagram."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    color_input = '#E8F4F8'      # Light blue for input
    color_audio = '#FFE5CC'       # Light orange for audio path
    color_video = '#CCE5FF'       # Light blue for video path
    color_fusion = '#E5CCFF'      # Light purple for fusion
    color_llm = '#FFCCCC'         # Light red for LLM
    color_output = '#CCFFCC'      # Light green for output
    
        # (Removed title text)
    
    # Input: Audiovisual Input
    input_box = FancyBboxPatch((3.5, 9.8), 3, 0.6, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_input, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 10.1, 'Audiovisual Input\n(synchronized A/V stream)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Split arrow
    ax.annotate('', xy=(3.5, 9), xytext=(5, 9.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(6.5, 9), xytext=(5, 9.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Audio Path (Left)
    # Audio Extraction
    audio_box1 = FancyBboxPatch((1, 8.2), 2, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=color_audio,
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(audio_box1)
    ax.text(2, 8.5, 'Audio Stream\nExtraction', ha='center', va='center', fontsize=9)
    
    # Arrow down
    ax.annotate('', xy=(2, 7.5), xytext=(2, 8.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # DeepGram ASR
    audio_box2 = FancyBboxPatch((0.8, 6.7), 2.4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=color_audio,
                                edgecolor='black', linewidth=2)
    ax.add_patch(audio_box2)
    ax.text(2, 7.3, 'Audio ASR', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2, 6.95, '(DeepGram)', ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow down
    ax.annotate('', xy=(2, 6.0), xytext=(2, 6.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Audio Output
    audio_box3 = FancyBboxPatch((0.5, 5.2), 3, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor=color_audio,
                                edgecolor='black', linewidth=1.5, linestyle='--')
    ax.add_patch(audio_box3)
    ax.text(2, 5.75, 'Y_a (transcript)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2, 5.4, 'C_a (word confidences)', ha='center', va='center', fontsize=8)
    
    # Video Path (Right)
    # Video Extraction
    video_box1 = FancyBboxPatch((7, 8.2), 2, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=color_video,
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(video_box1)
    ax.text(8, 8.5, 'Video Stream\nExtraction', ha='center', va='center', fontsize=9)
    
    # Arrow down
    ax.annotate('', xy=(8, 7.5), xytext=(8, 8.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # VSR Processing
    video_box2 = FancyBboxPatch((6.8, 6.7), 2.4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=color_video,
                                edgecolor='black', linewidth=2)
    ax.add_patch(video_box2)
    ax.text(8, 7.3, 'Visual SR', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8, 6.95, '(auto_avsr)', ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow down
    ax.annotate('', xy=(8, 6.0), xytext=(8, 6.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Video Output
    video_box3 = FancyBboxPatch((6.5, 5.2), 3, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor=color_video,
                                edgecolor='black', linewidth=1.5, linestyle='--')
    ax.add_patch(video_box3)
    ax.text(8, 5.75, 'Y_v (transcript)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(8, 5.4, 'C_v (word confidences)', ha='center', va='center', fontsize=8)
    
    # Convergence arrows to fusion
    ax.annotate('', xy=(4, 4.2), xytext=(2, 5.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(6, 4.2), xytext=(8, 5.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Confidence-aware Fusion
    fusion_box = FancyBboxPatch((3, 3.4), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=color_fusion,
                                edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 3.95, 'Confidence-Aware Fusion', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.6, '(word-level alignment)', ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow down
    ax.annotate('', xy=(5, 2.7), xytext=(5, 3.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # LLM Semantic Refinement
    llm_box = FancyBboxPatch((2.5, 1.7), 5, 1.0,
                             boxstyle="round,pad=0.1",
                             facecolor=color_llm,
                             edgecolor='black', linewidth=2.5)
    ax.add_patch(llm_box)
    ax.text(5, 2.4, 'LLM Semantic Refinement', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 2.05, '(Groq - constrained by confidence)', ha='center', va='center', fontsize=9, style='italic')
    ax.text(5, 1.85, 'Resolves disagreements, validates coherence', ha='center', va='center', fontsize=8)
    
    # Arrow down
    ax.annotate('', xy=(5, 1.0), xytext=(5, 1.7),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    
    # Final Output
    output_box = FancyBboxPatch((3, 0.2), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=color_output,
                                edgecolor='black', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(5, 0.75, 'Y* (Final Transcript)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 0.4, 'Semantically coherent, noise-robust', ha='center', va='center', fontsize=9, style='italic')
    
    # Add labels on the side
    ax.text(0.3, 7.5, 'Audio\nModality', ha='center', va='center', fontsize=10, 
            fontweight='bold', rotation=90, color='#CC6600')
    ax.text(9.7, 7.5, 'Video\nModality', ha='center', va='center', fontsize=10, 
            fontweight='bold', rotation=90, color='#0066CC')
    
        # (Removed caption text)
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure 3 saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_figure3("paper_figure3.png")
    print("✅ Figure 3 generated successfully!")
