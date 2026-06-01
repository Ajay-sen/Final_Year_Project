import numpy as np
import hashlib
import pywt
from scipy.integrate import odeint

class ImageEncryption:
    def __init__(self):
        # 5D Hyperchaotic System Parameters (Yang, 2009)
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

    def _generate_chaotic_sequences(self, length, x0):
        """Generate chaotic sequences of given length from initial conditions."""
        t = np.linspace(0, 500, length + 2000)
        states = odeint(self._hyperchaotic_system, x0, t)
        states = states[2000:]
        return states[:length]

    def _get_hash_initial_conditions(self, image_bytes):
        """Generates initial conditions (x0) from SHA3-512 hash of the image."""
        hash_obj = hashlib.sha3_512(image_bytes)
        hex_dig = hash_obj.hexdigest()
        
        x0 = []
        for i in range(5):
            chunk = hex_dig[i*8 : (i+1)*8]
            val = int(chunk, 16) / (2**32)
            x0.append(val + 0.1)
            
        return x0, hex_dig

    def _bwt_transform(self, data_bytes):
        """Burrows-Wheeler Transform - returns transformed data and index."""
        s = data_bytes
        block_size = min(len(s), 256)
        s = s[:block_size]
        n = block_size
        
        rotations = sorted(range(n), key=lambda i: s[i:] + s[:i])
        bwt_result = bytes([s[rotations[i] - 1] for i in range(n)])
        index = rotations.index(0)
        return bwt_result, index

    def _psm_encrypt(self, data, key):
        """Polyalphabetic Substitution Method with CBC feedback.
           C[i] = (P[i] + K[i] + C[i-1]) mod 256"""
        result = np.empty(len(data), dtype=np.uint8)
        prev = int(key[0])
        for i in range(len(data)):
            val = (int(data[i]) + int(key[i]) + prev) % 256
            result[i] = val
            prev = val
        return result

    def _psm_decrypt(self, data, key):
        """Inverse PSM with CBC.
           P[i] = (C[i] - K[i] - C[i-1]) mod 256"""
        result = np.empty(len(data), dtype=np.uint8)
        prev = int(key[0])
        for i in range(len(data)):
            result[i] = (int(data[i]) - int(key[i]) - prev) % 256
            prev = int(data[i])
        return result

    def _generate_key_stream(self, length, chaotic_states, state_idx):
        """Generate uniform key stream using rank-based quantization."""
        raw_seq = chaotic_states[:length, state_idx]
        ranks = np.argsort(np.argsort(raw_seq))
        key = (ranks * 256 // length).astype(np.uint8)
        return key

    def _wavelet_decompose(self, channel):
        """Decompose a channel into 4 sub-bands using pixel subsampling.
           Channel MUST have even dimensions (caller handles padding)."""
        h, w = channel.shape
        assert h % 2 == 0 and w % 2 == 0, "Channel must have even dimensions"
        
        LL = channel[0::2, 0::2].flatten()
        LH = channel[0::2, 1::2].flatten()
        HL = channel[1::2, 0::2].flatten()
        HH = channel[1::2, 1::2].flatten()
        
        return LL, LH, HL, HH

    def _wavelet_reconstruct(self, LL, LH, HL, HH, h, w):
        """Reconstruct channel from 4 sub-bands. Perfect inverse of decompose."""
        h2, w2 = h // 2, w // 2
        
        result = np.zeros((h, w), dtype=np.uint8)
        result[0::2, 0::2] = LL.reshape(h2, w2)
        result[0::2, 1::2] = LH.reshape(h2, w2)
        result[1::2, 0::2] = HL.reshape(h2, w2)
        result[1::2, 1::2] = HH.reshape(h2, w2)
        
        return result

    def encrypt(self, image_path):
        import cv2
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found")
        
        original_shape = img.shape
        h, w, channels = original_shape
        
        # Pad to even dimensions for lossless wavelet decomposition
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            img_padded = np.zeros((h + pad_h, w + pad_w, channels), dtype=np.uint8)
            img_padded[:h, :w, :] = img
            if pad_h:
                img_padded[h, :w, :] = img[h-1, :, :]
            if pad_w:
                img_padded[:h, w, :] = img[:, w-1, :]
            if pad_h and pad_w:
                img_padded[h, w, :] = img[h-1, w-1, :]
            img = img_padded
        
        shape = img.shape
        hp, wp, _ = shape
        flat = img.flatten()
        
        # 2. Key Generation (Plaintext Sensitive)
        x0, hash_hex = self._get_hash_initial_conditions(flat.tobytes())
        
        # BWT on hash for additional randomness
        bwt_data, bwt_index = self._bwt_transform(hash_hex.encode())
        
        # Generate random binary string (str) from BWT index
        np.random.seed(bwt_index % (2**31))
        random_str = np.random.randint(0, 256, size=32, dtype=np.uint8)
        
        # PSM on random string with BWT output to get STR_new
        bwt_key = np.frombuffer(bwt_data[:32].ljust(32, b'\x01'), dtype=np.uint8)
        str_new = self._psm_encrypt(random_str, bwt_key)
        
        # Modify initial conditions using str_new
        x0_modified = [x0[i] + (str_new[i] / 10000.0) for i in range(5)]
        
        # 3. Generate chaotic sequences for confusion
        states_conf = self._generate_chaotic_sequences(flat.size, x0_modified)
        
        # 4. Initial Scrambling (Confusion on full image)
        key_conf = states_conf[:flat.size, 0]
        sort_indices = np.argsort(key_conf)
        scrambled_flat = flat[sort_indices]
        scrambled_img = scrambled_flat.reshape(shape)
        
        # 5. Wavelet-domain Diffusion (per channel)
        encrypted_channels = []
        
        for ch in range(channels):
            channel = scrambled_img[:, :, ch]
            
            # Wavelet decomposition → 4 sub-bands (LL, LH, HL, HH)
            LL, LH, HL, HH = self._wavelet_decompose(channel)
            sub_bands = [LL, LH, HL, HH]
            
            encrypted_sub_bands = []
            
            for sb_idx, sb in enumerate(sub_bands):
                sb_size = len(sb)
                
                # Generate keys for this sub-band
                sb_seed = [x0_modified[j] + (ch * 4 + sb_idx + 1) * 0.00137 for j in range(5)]
                sb_states = self._generate_chaotic_sequences(sb_size, sb_seed)
                
                # Scramble sub-band
                scramble_key = sb_states[:, 2]
                sb_sort_idx = np.argsort(scramble_key)
                sb_scrambled = sb[sb_sort_idx]
                
                # PSM on sub-band
                psm_key = self._generate_key_stream(sb_size, sb_states, 1)
                sb_encrypted = self._psm_encrypt(sb_scrambled, psm_key)
                
                encrypted_sub_bands.append(sb_encrypted)
            
            # Random shuffle of 4 sub-band components
            shuffle_seed_val = int(np.abs(states_conf[ch * 100 + 50, 4]) * 10000) % (2**31)
            rng = np.random.RandomState(shuffle_seed_val)
            shuffle_order = rng.permutation(4)
            
            # Shuffle: position i gets sub-band shuffle_order[i]
            shuffled_sbs = [encrypted_sub_bands[shuffle_order[i]] for i in range(4)]
            
            # Reconstruct channel from shuffled encrypted sub-bands (IWLT)
            reconstructed = self._wavelet_reconstruct(
                shuffled_sbs[0], shuffled_sbs[1], shuffled_sbs[2], shuffled_sbs[3], hp, wp)
            
            encrypted_channels.append(reconstructed)
        
        # Stack channels
        encrypted_img_pre = np.stack(encrypted_channels, axis=2)
        
        # 6. Final PSM pass on full image
        enc_flat = encrypted_img_pre.flatten()
        final_seed = [x0_modified[i] + 0.00777 for i in range(5)]
        final_states = self._generate_chaotic_sequences(len(enc_flat), final_seed)
        final_psm_key = self._generate_key_stream(len(enc_flat), final_states, 0)
        encrypted_final = self._psm_encrypt(enc_flat, final_psm_key)
        
        encrypted_img = encrypted_final.reshape(shape)
        
        # Metadata for decryption
        metadata = {
            'x0_modified': x0_modified,
            'sort_indices': sort_indices,
            'shape': shape,
            'original_shape': original_shape,
        }
        
        return encrypted_img, metadata

    def decrypt(self, encrypted_img, metadata):
        import cv2
        
        x0_modified = metadata['x0_modified']
        sort_indices = metadata['sort_indices']
        shape = metadata['shape']
        original_shape = metadata['original_shape']
        hp, wp, channels = shape
        
        flat = encrypted_img.flatten()
        
        # 1. Inverse Final PSM
        final_seed = [x0_modified[i] + 0.00777 for i in range(5)]
        final_states = self._generate_chaotic_sequences(len(flat), final_seed)
        final_psm_key = self._generate_key_stream(len(flat), final_states, 0)
        dec_flat = self._psm_decrypt(flat, final_psm_key)
        
        dec_img = dec_flat.reshape(shape)
        
        # Regenerate confusion states for shuffle seeds
        states_conf = self._generate_chaotic_sequences(np.prod(shape), x0_modified)
        
        # 2. Per channel: WLT → unshuffle → inverse PSM → inverse scramble → IWLT
        decrypted_channels = []
        
        for ch in range(channels):
            channel = dec_img[:, :, ch]
            
            # Wavelet decomposition
            LL, LH, HL, HH = self._wavelet_decompose(channel)
            sub_bands_encrypted = [LL, LH, HL, HH]
            
            # Determine shuffle order (same seed as encryption)
            shuffle_seed_val = int(np.abs(states_conf[ch * 100 + 50, 4]) * 10000) % (2**31)
            rng = np.random.RandomState(shuffle_seed_val)
            shuffle_order = rng.permutation(4)
            
            # Unshuffle: during encryption, position i got sub-band shuffle_order[i]
            # So current position i contains original sub-band shuffle_order[i]
            unshuffled_sbs = [None] * 4
            for i in range(4):
                unshuffled_sbs[shuffle_order[i]] = sub_bands_encrypted[i]
            
            # Decrypt each sub-band: inverse PSM → inverse scramble
            decrypted_sub_bands = []
            
            for sb_idx in range(4):
                sb_encrypted = unshuffled_sbs[sb_idx]
                sb_size = len(sb_encrypted)
                
                # Regenerate same keys
                sb_seed = [x0_modified[j] + (ch * 4 + sb_idx + 1) * 0.00137 for j in range(5)]
                sb_states = self._generate_chaotic_sequences(sb_size, sb_seed)
                
                # Inverse PSM
                psm_key = self._generate_key_stream(sb_size, sb_states, 1)
                sb_decrypted = self._psm_decrypt(sb_encrypted, psm_key)
                
                # Inverse scramble
                scramble_key = sb_states[:, 2]
                sb_sort_idx = np.argsort(scramble_key)
                sb_unscrambled = np.zeros_like(sb_decrypted)
                sb_unscrambled[sb_sort_idx] = sb_decrypted
                
                decrypted_sub_bands.append(sb_unscrambled)
            
            # Inverse Wavelet Transform to reconstruct channel
            reconstructed = self._wavelet_reconstruct(
                decrypted_sub_bands[0], decrypted_sub_bands[1],
                decrypted_sub_bands[2], decrypted_sub_bands[3], hp, wp)
            
            decrypted_channels.append(reconstructed)
        
        # Stack channels
        decrypted_img = np.stack(decrypted_channels, axis=2)
        
        # 3. Inverse Confusion (un-shuffle full image)
        dec_flat = decrypted_img.flatten()
        result = np.zeros_like(dec_flat)
        result[sort_indices] = dec_flat
        
        # Reshape to padded size, then trim to original
        result_img = result.reshape(shape)
        return result_img[:original_shape[0], :original_shape[1], :]
