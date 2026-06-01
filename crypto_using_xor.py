import numpy as np
import hashlib
from scipy.integrate import odeint

class ImageEncryption:
    def __init__(self):
        # 5D Hyperchaotic System Parameters [Source: Maity et al.]
        self.a, self.b, self.c, self.d, self.e = 2.38, 3.0, 2.0, 15.8, 60.1

    def _hyperchaotic_system(self, state, t):
        x1, x2, x3, x4, x5 = state
        dx1 = -self.a*x1 - self.a*x2 + x5
        dx2 = self.b*x1 - self.c*x2 - x4
        dx3 = x2*x3 - x4
        dx4 = -self.d*x1 + x2*x4
        dx5 = self.e*x4 - self.c*x5
        return [dx1, dx2, dx3, dx4, dx5]

    def _generate_keys(self, image_shape, x0):
        """Generates chaotic sequences based on image dimensions."""
        total_pixels = image_shape[0] * image_shape[1]
        
        # Time steps for ODE solver
        t = np.linspace(0, 100, total_pixels + 1000) # Buffer for transient removal
        
        # Solve differential equations
        states = odeint(self._hyperchaotic_system, x0, t)
        
        # Discard first 1000 to remove transient effect
        states = states[1000:]
        
        # Normalize states to get usable keys
        # Key 1: For sorting/scrambling (Confusion)
        key_confusion = states[:total_pixels, 0] 
        
        # Key 2: For XOR operations (Diffusion) - Scale to 0-255
        key_diffusion = (np.abs(states[:total_pixels, 1]) * 1000).astype(int) % 256
        
        return key_confusion, key_diffusion.astype(np.uint8)

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

    def encrypt(self, image_path):
        import cv2
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Image not found")
        
        flat = img.flatten()
        shape = img.shape
        
        # 2. Key Generation (Plaintext Sensitive)
        x0 = self._get_hash_initial_conditions(flat.tobytes())
        key_conf, key_diff = self._generate_keys((flat.size, 1), x0)
        
        # 3. Confusion (Scrambling Pixel Positions)
        # We sort the chaotic sequence to get random indices
        sort_indices = np.argsort(key_conf)
        scrambled_flat = flat[sort_indices]
        
        # 4. Diffusion (Modifying Pixel Values)
        # XOR with the diffusion key
        encrypted_flat = np.bitwise_xor(scrambled_flat, key_diff)
        
        # Reshape back to image
        encrypted_img = encrypted_flat.reshape(shape)
        
        # Return necessary data for decryption
        # In a symmetric system, receiver generates keys from shared secret.
        # For this demo, we return x0 and indices to simulate having the key.
        return encrypted_img, x0, sort_indices, shape

    def decrypt(self, encrypted_img, x0, sort_indices, original_shape):
        flat = encrypted_img.flatten()
        
        # 1. Regenerate Keys (Using same x0)
        _, key_diff = self._generate_keys((flat.size, 1), x0)
        
        # 2. Inverse Diffusion (XOR is its own inverse)
        decrypted_diff = np.bitwise_xor(flat, key_diff)
        
        # 3. Inverse Confusion (Un-shuffling)
        # Create an empty array and place pixels back in original positions
        decrypted_conf = np.zeros_like(decrypted_diff)
        decrypted_conf[sort_indices] = decrypted_diff
        
        # Reshape
        return decrypted_conf.reshape(original_shape)