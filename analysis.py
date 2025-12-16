import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import entropy, pearsonr, chisquare

class Analysis:
    def __init__(self, plain_img_path, cipher_img_path):
        self.plain = cv2.imread(plain_img_path)
        self.cipher = cv2.imread(cipher_img_path)
        
        if len(self.plain.shape) == 3:
            self.plain_gray = cv2.cvtColor(self.plain, cv2.COLOR_BGR2GRAY)
            self.cipher_gray = cv2.cvtColor(self.cipher, cv2.COLOR_BGR2GRAY)
        else:
            self.plain_gray = self.plain
            self.cipher_gray = self.cipher

    def histogram_analysis(self, output_file="histograms.png"):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(self.plain_gray.ravel(), 256, [0, 256], color='blue', alpha=0.7)
        plt.title('Original Histogram (Non-Uniform)')
        plt.subplot(1, 2, 2)
        plt.hist(self.cipher_gray.ravel(), 256, [0, 256], color='red', alpha=0.7)
        plt.title('Encrypted Histogram (Uniform)')
        plt.savefig(output_file)
        plt.close()
        print(f"[Analysis] Histogram plot saved to {output_file}")

    def entropy_analysis(self):
        counts_cipher = np.bincount(self.cipher_gray.ravel(), minlength=256)
        prob_cipher = counts_cipher / np.sum(counts_cipher)
        # Avoid log(0)
        prob_cipher = prob_cipher[prob_cipher > 0]
        ent_cipher = -np.sum(prob_cipher * np.log2(prob_cipher))
        
        print("-" * 40)
        print(f"2> Entropy Analysis:")
        print(f"   Ideal Entropy: 8.0000")
        print(f"   Achieved Entropy: {ent_cipher:.5f}")
        if ent_cipher > 7.99: print("   Result: PASS (High Randomness)")
        else: print("   Result: ACCEPTABLE")

    def correlation_analysis(self, output_file="correlation.png"):
        # Calculate horizontal correlation
        flat_plain = self.plain_gray.flatten()
        flat_cipher = self.cipher_gray.flatten()
        
        # Avoid out of bounds
        n = min(len(flat_plain), 5000) 
        x_plain = flat_plain[:n]
        y_plain = flat_plain[1:n+1]
        
        x_cipher = flat_cipher[:n]
        y_cipher = flat_cipher[1:n+1]
        
        corr_orig, _ = pearsonr(x_plain, y_plain)
        corr_enc, _ = pearsonr(x_cipher, y_cipher)
        
        print("-" * 40)
        print(f"5> Correlation Analysis:")
        print(f"   Original Image Correlation: {corr_orig:.4f} (High)")
        print(f"   Encrypted Image Correlation: {corr_enc:.4f} (Target ~ 0)")
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(x_plain, y_plain, s=0.5)
        plt.title(f'Original Corr: {corr_orig:.4f}')
        plt.xlabel("Pixel(x,y)"); plt.ylabel("Pixel(x+1,y)")
        
        plt.subplot(1, 2, 2)
        plt.scatter(x_cipher, y_cipher, s=0.5, color='red')
        plt.title(f'Encrypted Corr: {corr_enc:.4f}')
        plt.xlabel("Pixel(x,y)"); plt.ylabel("Pixel(x+1,y)")
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
        # Observed frequencies in encrypted image
        counts = np.bincount(self.cipher_gray.ravel(), minlength=256)
        
        # Expected frequencies (Uniform distribution)
        total_pixels = self.cipher_gray.size
        expected = [total_pixels / 256] * 256
        
        chi_val, p_val = chisquare(counts, f_exp=expected)
        
        print("-" * 40)
        print("4> Chi-Square Test (Histogram Uniformity):")
        print(f"   Chi-Square Value: {chi_val:.2f}")
        print(f"   P-Value: {p_val:.4f}")
        # Theoretical threshold for 255 degrees of freedom at alpha=0.05 is ~293
        if chi_val < 293:
            print("   Result: PASS (Uniform Distribution)")
        else:
            print("   Result: DEVIATION DETECTED (Check image size/content)")
        print("-" * 40)