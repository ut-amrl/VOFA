import torch
import torch.nn as nn

class StateHistoryEncoder(nn.Module):
    """CNN-based temporal encoder for proprioceptive observation history."""
    
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super().__init__()
        self.activation_fn = activation_fn
        self.input_size = input_size
        self.tsteps = tsteps
        self._debug_printed = False
        
        # Build encoder architecture
        channel_size = 10
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), 
            self.activation_fn,
        )
        
        self.conv_layers = self._build_conv_layers(channel_size)
        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), 
            self.activation_fn
        )

    def _build_conv_layers(self, channel_size):
        """Build convolutional layers based on timestep configuration."""
        conv_configs = {
            50: [
                (3 * channel_size, 2 * channel_size, 8, 4),
                (2 * channel_size, channel_size, 5, 1),
                (channel_size, channel_size, 5, 1),
            ],
            20: [
                (3 * channel_size, 2 * channel_size, 6, 2),
                (2 * channel_size, channel_size, 4, 2),
            ],
            10: [
                (3 * channel_size, 2 * channel_size, 4, 2),
                (2 * channel_size, channel_size, 2, 1),
            ]
        }
        
        if self.tsteps not in conv_configs:
            raise ValueError(f"tsteps must be one of {list(conv_configs.keys())}")
        
        layers = []
        for in_ch, out_ch, kernel, stride in conv_configs[self.tsteps]:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel, stride),
                self.activation_fn,
            ])
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)

    def forward(self, obs):
        """
        Forward pass for history encoding.
        
        Args:
            obs: (B,L,D) or (H,N,L,D) observation tensor
            
        Returns:
            Encoded history features: (B, output_size) or (H*N, output_size)
        """
        obs = self._normalize_input_shape(obs)
        obs = self._validate_and_debug(obs)
        
        # Time-distributed projection
        batch_size, seq_len, feature_dim = obs.shape
        x = obs.contiguous().view(batch_size * seq_len, feature_dim)
        projection = self.encoder(x)
        projection = projection.view(batch_size, seq_len, -1)

        # Apply temporal convolutions
        output = self.conv_layers(projection.permute(0, 2, 1))
        output = self.linear_output(output)
        
        return output

    def _normalize_input_shape(self, obs):
        """Normalize input to 3D (B,L,D) format."""
        if obs.dim() == 4:
            H, N, L, D = obs.shape
            return obs.contiguous().view(H * N, L, D)
        elif obs.dim() == 3:
            return obs
        else:
            raise ValueError(f"Expected 3D/4D obs, got {obs.shape}")

    def _validate_and_debug(self, obs):
        """Validate input dimensions and print debug info once."""
        if not self._debug_printed:
            print(f"history encoder debug: obs.shape={obs.shape}, expected D={self.input_size}")
            self._debug_printed = True
        return obs
    
class StateHistoryCamEncoder(nn.Module):
    """CNN-based temporal encoder for proprioceptive observation history."""
    
    def __init__(self, activation_fn, input_size, tsteps, output_size, image_size, tanh_encoder_output=False):
        super().__init__()
        self.activation_fn = activation_fn
        self.input_size = input_size
        self.tsteps = tsteps
        self._debug_printed = False
        
        # Build encoder architecture
        half_channel_size = 10
        channel_size = half_channel_size * 2
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * half_channel_size), 
            self.activation_fn,
        )
        # TODO hardcoded: assume input images are grayscale 32x32
        self.image_size = image_size
        out_size = int(image_size // 8)  # after 3 conv layers with stride 2
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * out_size * out_size, 3 * half_channel_size),
            nn.ReLU(),
        )
        
        self.conv_layers = self._build_conv_layers(channel_size)
        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), 
            self.activation_fn
        )

    def _build_conv_layers(self, channel_size):
        """Build convolutional layers based on timestep configuration."""
        conv_configs = {
            50: [
                (3 * channel_size, 2 * channel_size, 8, 4),
                (2 * channel_size, channel_size, 5, 1),
                (channel_size, channel_size, 5, 1),
            ],
            20: [
                (3 * channel_size, 2 * channel_size, 6, 2),
                (2 * channel_size, channel_size, 4, 2),
            ],
            10: [
                (3 * channel_size, 2 * channel_size, 4, 2),
                (2 * channel_size, channel_size, 2, 1),
            ]
        }
        
        if self.tsteps not in conv_configs:
            raise ValueError(f"tsteps must be one of {list(conv_configs.keys())}")
        
        layers = []
        for in_ch, out_ch, kernel, stride in conv_configs[self.tsteps]:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel, stride),
                self.activation_fn,
            ])
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)

    def forward(self, obs):
        """
        Forward pass for history encoding.
        
        Args:
            obs: (B,L,D) or (H,N,L,D) observation tensor
            
        Returns:
            Encoded history features: (B, output_size) or (H*N, output_size)
        """
        obs = self._normalize_input_shape(obs)
        obs = self._validate_and_debug(obs)
        
        # Time-distributed projection
        batch_size, seq_len, feature_dim = obs.shape
        x = obs.contiguous().view(batch_size * seq_len, feature_dim)
        proprioception = x[:, :-self.image_size * self.image_size]  # assuming last part is image
        images = x[:, -self.image_size * self.image_size:].view(-1, 1, self.image_size, self.image_size)  # assuming grayscale images
        img_features = self.img_encoder(images)
        proprioception_features = self.encoder(proprioception)
        projection = torch.cat([proprioception_features, img_features], dim=-1)
        projection = projection.view(batch_size, seq_len, -1)

        # Apply temporal convolutions
        output = self.conv_layers(projection.permute(0, 2, 1))
        output = self.linear_output(output)
        
        return output

    def _normalize_input_shape(self, obs):
        """Normalize input to 3D (B,L,D) format."""
        if obs.dim() == 4:
            H, N, L, D = obs.shape
            return obs.contiguous().view(H * N, L, D)
        elif obs.dim() == 3:
            return obs
        else:
            raise ValueError(f"Expected 3D/4D obs, got {obs.shape}")

    def _validate_and_debug(self, obs):
        """Validate input dimensions and print debug info once."""
        if not self._debug_printed:
            print(f"history encoder debug: obs.shape={obs.shape}, expected D={self.input_size}")
            self._debug_printed = True
        return obs


class StudentCamActorCriticOld(nn.Module):
    """Student actor-critic network with history encoding and asymmetric observation access."""
    
    def __init__(self, num_act, num_prop, hist_len, num_privileged_obs, init_logstd=-2.0):
        super().__init__()
        self.num_latent = 64
        
        # Networks
        self.image_size= 32
        self.num_img_latent = 32
        num_prop = num_prop - self.image_size * self.image_size  # adjust for image size in proprioception
        self.hist_encoder = StateHistoryCamEncoder(nn.ELU(), num_prop, hist_len, self.num_latent, image_size=self.image_size)
        out_size = int(self.image_size // 8)  # after 3 conv layers with stride 2
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * out_size * out_size, self.num_img_latent),
            nn.ReLU(),
        )
        self.actor = self._build_actor_network(num_prop, num_act)
        self.critic = self._build_critic_network(num_prop, num_privileged_obs)
        self.logstd = nn.Parameter(torch.full((1, num_act), fill_value=init_logstd), requires_grad=False)
        
        self._print_network_info()

    def _build_actor_network(self, num_prop, num_act):
        """Build actor network (proprioceptive + history encoding only)."""
        return nn.Sequential(
            nn.Linear(num_prop + self.num_img_latent + self.num_latent, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, num_act),
        )

    def _build_critic_network(self, num_prop, num_privileged_obs):
        """Build critic network (proprioceptive + privileged observations)."""
        return nn.Sequential(
            nn.Linear(num_prop + num_privileged_obs, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def _print_network_info(self):
        """Print network architecture information."""
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def act(self, obs, privileged_obs=None, hist_encoding=True):
        """
        Generate action distribution.
        
        Args:
            obs: (N,L,D) or (H,N,L,D) observation tensor
            privileged_obs: Optional privileged observations
            hist_encoding: Whether to use history encoder. Ture for rollouts, false for BC.
            
        Returns:
            torch.distributions.Normal: Action distribution with proper shape
        """
        # Handle shape normalization
        shape_info = self._parse_input_shape(obs, privileged_obs)
        obs_flat = shape_info['obs_flat']
        
        # Get latent features
        latent = self._get_latent_features(obs_flat, hist_encoding)
        
        # Compute action distribution
        last_obs = obs_flat[:, -1, :]
        proprioception = last_obs[:, :-self.image_size * self.image_size]  # assuming last part is image
        image = last_obs[:, -self.image_size * self.image_size:].view(-1, 1, self.image_size, self.image_size)  # assuming grayscale images
        img_latent = self.img_encoder(image)
        feats = torch.cat([proprioception, img_latent, latent], dim=-1)
        loc = self.actor(feats)
        mean = torch.tanh(loc)
        scale = torch.exp(self.logstd).expand_as(mean)
        
        # Reshape back to original format if needed
        if shape_info['was_4d']:
            mean, scale = self._reshape_outputs_4d(mean, scale, shape_info)
            self._debug_assert_action_shape(mean, shape_info)
        
        return torch.distributions.Normal(mean, scale)

    def est_value(self, obs, privileged_obs):
        
        # Estimate state values.
        
        # shape normalization
        shape_info = self._parse_input_shape(obs, privileged_obs)
        obs_flat = shape_info['obs_flat']
        priv_flat = shape_info['priv_flat']
        
        last_obs = obs_flat[:, -1, :]
        critic_input = torch.cat([last_obs, priv_flat], dim=-1)
        values = self.critic(critic_input).squeeze(-1)
        
        # reshape back to original if needed
        if shape_info['was_4d']:
            values = self._reshape_values_4d(values, shape_info)
            self._debug_assert_value_shape(values, shape_info)
        
        return values

    def _parse_input_shape(self, obs, privileged_obs):
        """Parse and normalize input tensor shapes."""
        was_4d = (obs.dim() == 4)
        
        if was_4d:
            H, N, L, D = obs.shape
            obs_flat = obs.contiguous().view(H * N, L, D)
            priv_flat = privileged_obs.contiguous().view(H * N, -1) if privileged_obs is not None else None
        elif obs.dim() == 3:
            N, L, D = obs.shape
            H = None
            obs_flat = obs
            priv_flat = privileged_obs
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")
        
        return {
            'was_4d': was_4d,
            'H': H, 'N': N, 'L': L, 'D': D,
            'obs_flat': obs_flat,
            'priv_flat': priv_flat
        }

    def _get_latent_features(self, obs, hist_encoding):
        # Get latent features
        if hist_encoding:
            return self.hist_encoder(obs)
        else:
            # stateless path
            return torch.zeros(obs.size(0), self.num_latent, device=obs.device, dtype=obs.dtype)

    def _reshape_outputs_4d(self, loc, scale, shape_info):
        """Reshape actor outputs back to 4D format."""
        H, N = shape_info['H'], shape_info['N']
        loc = loc.view(H, N, -1)
        scale = scale.view(H, N, -1)
        return loc, scale

    def _reshape_values_4d(self, values, shape_info):
        """Reshape critic values back to 4D format."""
        H, N = shape_info['H'], shape_info['N']
        return values.view(H, N)

    def _debug_assert_action_shape(self, loc, shape_info):
        """Debug assertion for action shape (can be removed later)."""
        H, N = shape_info['H'], shape_info['N']
        expected_actions = getattr(self, 'num_actions', loc.shape[-1])
        assert loc.shape == (H, N, expected_actions), \
            f"Expected (H={H}, N={N}, A={expected_actions}), got {loc.shape}"

    def _debug_assert_value_shape(self, values, shape_info):
        """Debug assertion for value shape (can be removed later)."""
        H, N = shape_info['H'], shape_info['N']
        assert values.shape == (H, N), f"Expected (H={H}, N={N}), got {values.shape}"

class StudentCamActorCritic(nn.Module):
    """Student actor-critic network with history encoding and asymmetric observation access."""
    
    def __init__(self, num_act, num_prop, hist_len, num_privileged_obs, init_logstd=-2.0):
        super().__init__()
        self.num_latent = 64
        
        # Networks
        self.image_size= 32 # hardcoded
        self.depth_buffer = 3 # hardcoded
        self.num_img_latent = 32
        # self.img_seq_latent = 32
        out_size = int(self.image_size // 8)  # after 3 conv layers with stride 2
        
        self.hist_encoder = StateHistoryEncoder(nn.ELU(), num_prop, hist_len, self.num_latent)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), # 3 x 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * out_size * out_size, self.num_img_latent),
            nn.ReLU(),
        )
        # self.img_seq_proj = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.depth_buffer * self.num_img_latent, self.img_seq_latent),
        #     nn.ReLU(),
        #     nn.Linear(self.img_seq_latent, self.img_seq_latent),
        #     nn.ReLU(),
        # )
        self.actor = self._build_actor_network(num_prop, num_act)
        self.critic = self._build_critic_network(num_prop, num_privileged_obs)
        self.logstd = nn.Parameter(torch.full((1, num_act), fill_value=init_logstd), requires_grad=False)
        
        self._print_network_info()

    def _build_actor_network(self, num_prop, num_act):
        """Build actor network (proprioceptive + history encoding only)."""
        return nn.Sequential(
            nn.Linear(num_prop + self.num_img_latent + self.num_latent, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, num_act),
        )

    def _build_critic_network(self, num_prop, num_privileged_obs):
        """Build critic network (proprioceptive + privileged observations)."""
        return nn.Sequential(
            nn.Linear(num_prop + num_privileged_obs, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def _print_network_info(self):
        """Print network architecture information."""
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def act(self, obs, depth_obs, privileged_obs=None, hist_encoding=True):
        """
        Generate action distribution.
        
        Args:
            obs: (N,L,D) or (H,N,L,D) observation tensor
            privileged_obs: Optional privileged observations
            hist_encoding: Whether to use history encoder. Ture for rollouts, false for BC.
            
        Returns:
            torch.distributions.Normal: Action distribution with proper shape
        """
        # Handle shape normalization
        shape_info = self._parse_input_shape(obs, privileged_obs)
        obs_flat = shape_info['obs_flat']
        
        # Get latent features
        latent = self._get_latent_features(obs_flat, hist_encoding)
        
        # Compute action distribution
        proprioception = obs_flat[:, -1, :]
        B = depth_obs.shape[0]
        
        # # plot this depth image for debug
        # if depth_obs.shape[0] == 1:
        #     for i in range(depth_obs.shape[1]):
        #         # depth_obs: (B, C, H, W) → take channel i
        #         depth_img = depth_obs[0, i].cpu().numpy()     # now shape: (H, W)

        #         # normalize to 0–255
        #         depth_img = (depth_img + 0.5) * 255
        #         depth_uint8 = depth_img.astype(np.uint8)

        #         print("Depth Image min/max:", depth_uint8.min(), depth_uint8.max(), depth_uint8.shape)

        #         # Resize
        #         depth_resized = cv2.resize(depth_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)

        #         # Apply colormap (must be HxW uint8)
        #         colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_JET)

        #         cv2.imwrite(f"depth_image_{i}.png", colored)

        #     input("Save Images, Press Enter to continue..."  )
        # depth_obs = depth_obs.view(B * self.depth_buffer, 1, self.image_size, self.image_size)
        img_latent = self.img_encoder(depth_obs)
        # img_latent = img_latent.view(B, self.depth_buffer, self.num_img_latent)
        # img_seq_latent = self.img_seq_proj(img_latent)
        # img_latent = img_latent[:, -1, :]

        feats = torch.cat([proprioception, img_latent, latent], dim=-1)
        loc = self.actor(feats)
        mean = torch.tanh(loc)
        scale = torch.exp(self.logstd).expand_as(mean)
        
        # Reshape back to original format if needed
        if shape_info['was_4d']:
            mean, scale = self._reshape_outputs_4d(mean, scale, shape_info)
            self._debug_assert_action_shape(mean, shape_info)
        
        return torch.distributions.Normal(mean, scale)

    def est_value(self, obs, privileged_obs):
        
        # Estimate state values.
        
        # shape normalization
        shape_info = self._parse_input_shape(obs, privileged_obs)
        obs_flat = shape_info['obs_flat']
        priv_flat = shape_info['priv_flat']
        
        last_obs = obs_flat[:, -1, :]
        critic_input = torch.cat([last_obs, priv_flat], dim=-1)
        values = self.critic(critic_input).squeeze(-1)
        
        # reshape back to original if needed
        if shape_info['was_4d']:
            values = self._reshape_values_4d(values, shape_info)
            self._debug_assert_value_shape(values, shape_info)
        
        return values

    def _parse_input_shape(self, obs, privileged_obs):
        """Parse and normalize input tensor shapes."""
        N, L, D = obs.shape
        H = None
        obs_flat = obs
        priv_flat = privileged_obs
        
        return {
            'was_4d': False,
            'H': H, 'N': N, 'L': L, 'D': D,
            'obs_flat': obs_flat,
            'priv_flat': priv_flat
        }

    def _get_latent_features(self, obs, hist_encoding):
        # Get latent features
        return self.hist_encoder(obs)

    def _reshape_outputs_4d(self, loc, scale, shape_info):
        """Reshape actor outputs back to 4D format."""
        H, N = shape_info['H'], shape_info['N']
        loc = loc.view(H, N, -1)
        scale = scale.view(H, N, -1)
        return loc, scale

    def _reshape_values_4d(self, values, shape_info):
        """Reshape critic values back to 4D format."""
        H, N = shape_info['H'], shape_info['N']
        return values.view(H, N)

    def _debug_assert_action_shape(self, loc, shape_info):
        """Debug assertion for action shape (can be removed later)."""
        H, N = shape_info['H'], shape_info['N']
        expected_actions = getattr(self, 'num_actions', loc.shape[-1])
        assert loc.shape == (H, N, expected_actions), \
            f"Expected (H={H}, N={N}, A={expected_actions}), got {loc.shape}"

    def _debug_assert_value_shape(self, values, shape_info):
        """Debug assertion for value shape (can be removed later)."""
        H, N = shape_info['H'], shape_info['N']
        assert values.shape == (H, N), f"Expected (H={H}, N={N}), got {values.shape}"
        
class StudentCamActorCriticNew(nn.Module):
    """Student actor-critic network with history encoding and asymmetric observation access."""
    
    def __init__(self, num_act, num_prop, hist_len, num_privileged_obs, init_logstd=-2.0):
        super().__init__()
        self.num_latent = 64
        
        # Networks
        self.image_size= 32 # hardcoded
        self.depth_buffer = 3 # hardcoded
        self.num_img_latent = 32
        # self.img_seq_latent = 32
        out_size = int(self.image_size // 8)  # after 3 conv layers with stride 2
        
        self.hist_encoder = StateHistoryEncoder(nn.ELU(), num_prop, hist_len, self.num_latent)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1), # 1 x 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * out_size * out_size, self.num_img_latent),
            nn.ReLU(),
        )
        self.img_seq_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.depth_buffer * self.num_img_latent, self.num_img_latent),
            nn.ReLU(),
            nn.Linear(self.num_img_latent, self.num_img_latent),
            nn.ReLU(),
        )
        self.actor = self._build_actor_network(num_prop, num_act)
        self.critic = self._build_critic_network(num_prop, num_privileged_obs)
        self.logstd = nn.Parameter(torch.full((1, num_act), fill_value=init_logstd), requires_grad=False)
        
        self._print_network_info()

    def _build_actor_network(self, num_prop, num_act):
        """Build actor network (proprioceptive + history encoding only)."""
        return nn.Sequential(
            nn.Linear(num_prop + self.num_img_latent + self.num_latent, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, num_act),
        )

    def _build_critic_network(self, num_prop, num_privileged_obs):
        """Build critic network (proprioceptive + privileged observations)."""
        return nn.Sequential(
            nn.Linear(num_prop + num_privileged_obs, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def _print_network_info(self):
        """Print network architecture information."""
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        
    @torch.jit.export
    def forward(self, obs, depth_obs):
        latent = self.hist_encoder(obs)
        # Compute action distribution
        proprioception = obs[:, -1, :]
        B = depth_obs.shape[0]
        
        depth_obs = depth_obs.view(B * self.depth_buffer, 1, self.image_size, self.image_size)
        img_latent = self.img_encoder(depth_obs)
        img_latent = img_latent.view(B, self.depth_buffer, self.num_img_latent)
        img_latent = self.img_seq_proj(img_latent)

        feats = torch.cat([proprioception, img_latent, latent], dim=-1)
        loc = self.actor(feats)
        mean = torch.tanh(loc)
        return mean

    def act(self, obs, depth_obs, privileged_obs=None, hist_encoding=True):
        """
        Generate action distribution.
        
        Args:
            obs: (N,L,D) or (H,N,L,D) observation tensor
            privileged_obs: Optional privileged observations
            hist_encoding: Whether to use history encoder. Ture for rollouts, false for BC.
            
        Returns:
            torch.distributions.Normal: Action distribution with proper shape
        """
        # Handle shape normalization
        shape_info = self._parse_input_shape(obs, privileged_obs)
        obs_flat = shape_info['obs_flat']
        
        # Get latent features
        latent = self._get_latent_features(obs_flat, hist_encoding)
        
        # Compute action distribution
        proprioception = obs_flat[:, -1, :]
        if shape_info['was_4d']:
            B = depth_obs.shape[0] * depth_obs.shape[1] # H * N
        else:
            B = depth_obs.shape[0]
        
        # # plot this depth image for debug
        # if depth_obs.shape[0] == 1:
        #     for i in range(depth_obs.shape[1]):
        #         # depth_obs: (B, C, H, W) → take channel i
        #         depth_img = depth_obs[0, i].cpu().numpy()     # now shape: (H, W)

        #         # normalize to 0–255
        #         depth_img = (depth_img + 0.5) * 255
        #         depth_uint8 = depth_img.astype(np.uint8)

        #         print("Depth Image min/max:", depth_uint8.min(), depth_uint8.max(), depth_uint8.shape)

        #         # Resize
        #         depth_resized = cv2.resize(depth_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)

        #         # Apply colormap (must be HxW uint8)
        #         colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_JET)

        #         cv2.imwrite(f"depth_image_{i}.png", colored)

        #     input("Save Images, Press Enter to continue..."  )
        depth_obs = depth_obs.view(B * self.depth_buffer, 1, self.image_size, self.image_size)
        img_latent = self.img_encoder(depth_obs)
        img_latent = img_latent.view(B, self.depth_buffer, self.num_img_latent)
        img_latent = self.img_seq_proj(img_latent)
        # img_latent = img_latent[:, -1, :]

        feats = torch.cat([proprioception, img_latent, latent], dim=-1)
        loc = self.actor(feats)
        mean = torch.tanh(loc)
        scale = torch.exp(self.logstd).expand_as(mean)
        
        # Reshape back to original format if needed
        if shape_info['was_4d']:
            mean, scale = self._reshape_outputs_4d(mean, scale, shape_info)
            self._debug_assert_action_shape(mean, shape_info)
        
        return torch.distributions.Normal(mean, scale)

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs[..., -1, :], privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)
        
    def _parse_input_shape(self, obs, privileged_obs):
        """Parse and normalize input tensor shapes."""
        was_4d = (obs.dim() == 4)
        
        if was_4d:
            H, N, L, D = obs.shape
            obs_flat = obs.contiguous().view(H * N, L, D)
            priv_flat = privileged_obs.contiguous().view(H * N, -1) if privileged_obs is not None else None
        elif obs.dim() == 3:
            N, L, D = obs.shape
            H = None
            obs_flat = obs
            priv_flat = privileged_obs
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")
        
        return {
            'was_4d': was_4d,
            'H': H, 'N': N, 'L': L, 'D': D,
            'obs_flat': obs_flat,
            'priv_flat': priv_flat
        }

    def _get_latent_features(self, obs, hist_encoding):
        # Get latent features
        return self.hist_encoder(obs)

    def _reshape_outputs_4d(self, loc, scale, shape_info):
        """Reshape actor outputs back to 4D format."""
        H, N = shape_info['H'], shape_info['N']
        loc = loc.view(H, N, -1)
        scale = scale.view(H, N, -1)
        return loc, scale

    def _reshape_values_4d(self, values, shape_info):
        """Reshape critic values back to 4D format."""
        H, N = shape_info['H'], shape_info['N']
        return values.view(H, N)

    def _debug_assert_action_shape(self, loc, shape_info):
        """Debug assertion for action shape (can be removed later)."""
        H, N = shape_info['H'], shape_info['N']
        expected_actions = getattr(self, 'num_actions', loc.shape[-1])
        assert loc.shape == (H, N, expected_actions), \
            f"Expected (H={H}, N={N}, A={expected_actions}), got {loc.shape}"

    def _debug_assert_value_shape(self, values, shape_info):
        """Debug assertion for value shape (can be removed later)."""
        H, N = shape_info['H'], shape_info['N']
        assert values.shape == (H, N), f"Expected (H={H}, N={N}), got {values.shape}"
        
class StudentCamActorCriticNewPP(StudentCamActorCriticNew):
    def __init__(self, num_act, num_prop, hist_len, num_privileged_obs, init_logstd=-2.0, num_critic_prop=None):
        self.num_critic_prop = num_critic_prop if num_critic_prop is not None else num_prop
        super().__init__(num_act, num_prop, hist_len, num_privileged_obs, init_logstd)

    def _build_critic_network(self, num_prop, num_privileged_obs):
            """Build critic network (proprioceptive + privileged observations)."""
            return nn.Sequential(
                nn.Linear(self.num_critic_prop + num_privileged_obs, 256), nn.ELU(),
                nn.Linear(256, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 1),
            )