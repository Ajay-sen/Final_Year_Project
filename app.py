import cv2
import numpy as np
from crypto import ImageEncryption
from analysis import Analysis
import os

def main():
    # ---------------- SETTINGS ----------------
    image_path = 'mypic.jpeg'  # Make sure this file exists!
    # ------------------------------------------

    if not os.path.exists(image_path):
        print(f"ERROR: File '{image_path}' not found.")
        return

    print("\n=== PROJECT SEMINAR: IMAGE ENCRYPTION USING 5D HYPERCHAOS ===\n")

    print("--- STEP 1: ENCRYPTION ---")
    encryptor = ImageEncryption()
    
    # Encrypt
    try:
        cipher_img, metadata = encryptor.encrypt(image_path)
    except Exception as e:
        print(f"Encryption Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save as PNG to prevent compression artifacts affecting analysis
    cv2.imwrite('encrypted.png', cipher_img)
    print(f"Success! Encrypted image saved as 'encrypted.png'.")

    print("\n--- STEP 2: DECRYPTION ---")
    # Load the PNG we just saved
    enc_loaded = cv2.imread('encrypted.png')
    
    # Decrypt
    decrypted_img = encryptor.decrypt(enc_loaded, metadata)
    
    # Save Result
    cv2.imwrite('decrypted.png', decrypted_img)
    print("Success! Decrypted image saved as 'decrypted.png'.")
    
    print("\n--- STEP 3: SECURITY ANALYSIS RESULTS ---")
    # Initialize Analyzer
    analyzer = Analysis(image_path, 'encrypted.png')
    
    # 1. Key Space
    analyzer.key_space_analysis()
    
    # 2. Entropy
    analyzer.entropy_analysis()
    
    # 3. Histogram
    analyzer.histogram_analysis()
    
    # 4. Chi Square
    analyzer.chi_square_test()
    
    # 5. Correlation
    analyzer.correlation_analysis()
    
    print("\nAll analysis tasks completed.")
    print("Generated Plots: histograms.png, correlation.png")
    print("Decrypted Image: decrypted.png")

if __name__ == "__main__":
    main()