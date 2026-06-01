import numpy as np
import hashlib
from scipy.integrate import odeint

class ImageEncryption:
    def __init__(self):
        # 5D Hyperchaotic System Parameters (Lorenz-based, Yang 2009)
        # This system is numerically stable and produces high-quality chaotic sequences
        self.a = 10
        self.b = 8/3
        self.c = 28
        self.d = 2
        self.e = 1

    def _hyperchaotic_system(self, state, t):
        """5D Hyperchaotic system equations."""
        x1, x2, x3, x4, x5 = state
        dx1 = self.a * (x2 - x1) + x4
        dx2 = self.c * x1 - x1 * x3 + x5
        dx3 = x1 * x2 - self.b * x3
        dx4 = -self.d * x1
        dx5 = -self.e * x2
        return [dx1, dx2, dx3, dx4, dx5]

    def _generate_keys(self, total_pixels, x0):
        """Generates chaotic sequences based on total pixel count."""
        # Time steps for ODE solver
        t = np.linspace(0, 500, total_pixels + 2000)
        
        # Solve differential equations
        states = odeint(self._hyperchaotic_system, x0, t)
        
        # Discard first 2000 to remove transient effect
        states = states[2000:]
        
        # Key 1: For sorting/scrambling (Confusion) - use state variable x1
        key_confusion = states[:total_pixels, 0]
        
        # Key 2: For Diffusion - use rank-based quantization for guaranteed uniformity
        # Rank-based: sort the sequence, assign ranks, map ranks to 0-255
        raw_seq = states[:total_pixels, 1]
        ranks = np.argsort(np.argsort(raw_seq))  # Double argsort gives ranks
        key_diffusion = (ranks * 256 // total_pixels).astype(np.uint8)
        
        # Key 3: Second diffusion key from another state variable
        raw_seq2 = states[:total_pixels, 3]
        ranks2 = np.argsort(np.argsort(raw_seq2))
        key_diffusion2 = (ranks2 * 256 // total_pixels).astype(np.uint8)
        
        return key_confusion, key_diffusion, key_diffusion2

    def _get_hash_initial_conditions(self, image_bytes):
        """Generates initial conditions (x0) from SHA-512 hash of the image."""
        hash_obj = hashlib.sha512(image_bytes)
        hex_dig = hash_obj.hexdigest()
        
        # Convert distinct parts of hash to float initial conditions
        # We need 5 initial values for 5D system
        x0 = []
        for i in range(5):
            # Take slices of the hash, convert hex to int, normalize
            chunk = hex_dig[i*8 : (i+1)*8]
            val = int(chunk, 16) / (2**32) # Normalize to 0-1 range roughly
            x0.append(val + 0.1) # Avoid zero state
            
        return x0

    def _diffuse_forward(self, data, key):
        """CBC-style forward diffusion: C[i] = (P[i] + K[i] + C[i-1]) mod 256"""
        result = np.empty(len(data), dtype=np.uint8)
        prev = int(key[0])  # IV
        for i in range(len(data)):
            val = (int(data[i]) + int(key[i]) + prev) % 256
            result[i] = val
            prev = val
        return result

    def _diffuse_backward(self, data, key):
        """CBC-style backward diffusion: C[i] = (P[i] + K[i] + C[i+1]) mod 256"""
        result = np.empty(len(data), dtype=np.uint8)
        prev = int(key[-1])  # IV
        for i in range(len(data) - 1, -1, -1):
            val = (int(data[i]) + int(key[i]) + prev) % 256
            result[i] = val
            prev = val
        return result

    def _inverse_diffuse_forward(self, data, key):
        """Inverse of forward CBC diffusion."""
        result = np.empty(len(data), dtype=np.uint8)
        prev = int(key[0])
        for i in range(len(data)):
            result[i] = (int(data[i]) - int(key[i]) - prev) % 256
            prev = int(data[i])
        return result

    def _inverse_diffuse_backward(self, data, key):
        """Inverse of backward CBC diffusion."""
        result = np.empty(len(data), dtype=np.uint8)
        prev = int(key[-1])
        for i in range(len(data) - 1, -1, -1):
            result[i] = (int(data[i]) - int(key[i]) - prev) % 256
            prev = int(data[i])
        return result

    def encrypt(self, image_path):
        import cv2
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Image not found")
        
        flat = img.flatten()
        shape = img.shape
        
        # 2. Key Generation (Plaintext Sensitive)
        x0 = self._get_hash_initial_conditions(flat.tobytes())
        key_conf, key_diff1, key_diff2 = self._generate_keys(flat.size, x0)
        
        # 3. Confusion (Scrambling Pixel Positions)
        sort_indices = np.argsort(key_conf)
        scrambled_flat = flat[sort_indices]
        
        # 4. Diffusion with CBC-style feedback (two passes for full propagation)
        # Forward pass with key1: each C[i] depends on C[i-1]
        diffused = self._diffuse_forward(scrambled_flat, key_diff1)
        # Backward pass with key2: each C[i] depends on C[i+1]
        encrypted_flat = self._diffuse_backward(diffused, key_diff2)
        
        # Reshape back to image
        encrypted_img = encrypted_flat.reshape(shape)
        
        return encrypted_img, x0, sort_indices, shape

    def decrypt(self, encrypted_img, x0, sort_indices, original_shape):
        flat = encrypted_img.flatten()
        
        # 1. Regenerate Keys (Using same x0)
        _, key_diff1, key_diff2 = self._generate_keys(flat.size, x0)
        
        # 2. Inverse Diffusion (reverse order: backward first, then forward)
        inv_backward = self._inverse_diffuse_backward(flat, key_diff2)
        decrypted_diff = self._inverse_diffuse_forward(inv_backward, key_diff1)
        
        # 3. Inverse Confusion (Un-shuffling)
        decrypted_conf = np.zeros_like(decrypted_diff)
        decrypted_conf[sort_indices] = decrypted_diff
        
        # Reshape
        return decrypted_conf.reshape(original_shape)
