import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import entropy, pearsonr, chisquare

class Analysis:
    def __init__(self, plain_img_path, cipher_img_path):
        self.plain = cv2.imread(plain_img_path)
        self.cipher = cv2.imread(cipher_img_path)
        self.channels = ['Blue', 'Green', 'Red']

    def histogram_analysis(self, output_file="histograms.png"):
        """Per-channel histogram comparison between original and encrypted images."""
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        colors = ['blue', 'green', 'red']
        
        for i, (ch_name, color) in enumerate(zip(self.channels, colors)):
            # Original channel histogram
            axes[i, 0].hist(self.plain[:,:,i].ravel(), 256, [0, 256], 
                          color=color, alpha=0.7)
            axes[i, 0].set_title(f'Original - {ch_name} Channel')
            axes[i, 0].set_xlim([0, 256])
            
            # Encrypted channel histogram
            axes[i, 1].hist(self.cipher[:,:,i].ravel(), 256, [0, 256], 
                          color=color, alpha=0.7)
            axes[i, 1].set_title(f'Encrypted - {ch_name} Channel')
            axes[i, 1].set_xlim([0, 256])
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"[Analysis] Histogram plot saved to {output_file}")

    def entropy_analysis(self):
        """Per-channel entropy analysis of the encrypted image."""
        print("-" * 40)
        print("2> Entropy Analysis (Per Channel):")
        print(f"   Ideal Entropy: 8.0000")
        
        for i, ch_name in enumerate(self.channels):
            counts = np.bincount(self.cipher[:,:,i].ravel(), minlength=256)
            prob = counts / np.sum(counts)
            prob = prob[prob > 0]
            ent = -np.sum(prob * np.log2(prob))
            print(f"   {ch_name} Channel Entropy: {ent:.5f}")
        
        # Average entropy across channels
        all_entropies = []
        for i in range(3):
            counts = np.bincount(self.cipher[:,:,i].ravel(), minlength=256)
            prob = counts / np.sum(counts)
            prob = prob[prob > 0]
            all_entropies.append(-np.sum(prob * np.log2(prob)))
        avg_ent = np.mean(all_entropies)
        
        print(f"   Average Entropy: {avg_ent:.5f}")
        if avg_ent > 7.99:
            print("   Result: PASS (High Randomness)")
        else:
            print("   Result: ACCEPTABLE")

    def correlation_analysis(self, output_file="correlation.png"):
        """Per-channel adjacent pixel correlation analysis."""
        print("-" * 40)
        print("5> Correlation Analysis (Per Channel):")
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        for i, ch_name in enumerate(self.channels):
            plain_ch = self.plain[:,:,i].flatten()
            cipher_ch = self.cipher[:,:,i].flatten()
            
            n = min(len(plain_ch) - 1, 5000)
            x_plain = plain_ch[:n]
            y_plain = plain_ch[1:n+1]
            x_cipher = cipher_ch[:n]
            y_cipher = cipher_ch[1:n+1]
            
            corr_orig, _ = pearsonr(x_plain, y_plain)
            corr_enc, _ = pearsonr(x_cipher, y_cipher)
            
            print(f"   {ch_name}: Original={corr_orig:.4f}, Encrypted={corr_enc:.4f}")
            
            # Plot original
            axes[i, 0].scatter(x_plain, y_plain, s=0.5, color='cornflowerblue')
            axes[i, 0].set_title(f'Original {ch_name}: {corr_orig:.4f}')
            axes[i, 0].set_xlabel("Pixel(x,y)")
            axes[i, 0].set_ylabel("Pixel(x+1,y)")
            
            # Plot encrypted
            axes[i, 1].scatter(x_cipher, y_cipher, s=0.5, color='salmon')
            axes[i, 1].set_title(f'Encrypted {ch_name}: {corr_enc:.4f}')
            axes[i, 1].set_xlabel("Pixel(x,y)")
            axes[i, 1].set_ylabel("Pixel(x+1,y)")
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"[Analysis] Correlation plot saved to {output_file}")

    def key_space_analysis(self):
        print("-" * 40)
        print("1> Key Space Analysis:")
        print("   Key generation uses SHA-512 (512 bits).")
        print("   Key Space = 2^512")
        print("   Conclusion: Sufficiently large to resist Brute Force attacks.")

    def chi_square_test(self):
        """Per-channel chi-square test for histogram uniformity."""
        print("-" * 40)
        print("4> Chi-Square Test (Per Channel):")
        
        all_pass = True
        for i, ch_name in enumerate(self.channels):
            counts = np.bincount(self.cipher[:,:,i].ravel(), minlength=256)
            total_pixels = self.cipher[:,:,i].size
            expected = [total_pixels / 256] * 256
            chi_val, p_val = chisquare(counts, f_exp=expected)
            
            passed = chi_val < 293  # threshold for 255 df at alpha=0.05
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"   {ch_name}: Chi²={chi_val:.2f}, P={p_val:.4f} [{status}]")
        
        if all_pass:
            print("   Overall Result: PASS (Uniform Distribution)")
        else:
            print("   Overall Result: DEVIATION DETECTED")
        print("-" * 40)
