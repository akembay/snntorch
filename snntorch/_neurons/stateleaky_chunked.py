"""
StateLeaky implementation with chunking and conv optimizations - with batch (& channel) chunking 
"""
import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile
from neurons import LIF
import time


class StateLeaky_chunking(LIF):
    """StateLeaky with chunking optimizations.
    
    Parameters:
    - enable_chunking: switch for all chunking (default: True)
    - enable_batch_chunking: Enable batch-wise chunking (default: True)
    - enable_channel_chunking: Enable channel-wise chunking (default: True)
    
    When enable_chunking=False: Uses original StateLeaky implementation
    When enable_chunking=True: Uses selected chunking strategies
    """
    
    def __init__(
        self, 
        beta, 
        channels, 
        threshold=1.0, 
        spike_grad=None, 
        surrogate_disable=False,
        learn_beta=False, 
        learn_threshold=False, 
        state_quant=False, 
        output=True,
        graded_spikes_factor=1.0, 
        learn_graded_spikes_factor=False,
        enable_chunking=True,           # Enable/disable all chunking functionality
        enable_batch_chunking=True,     # Enable/disable batch chunking specifically
        enable_channel_chunking=True,   # Enable/disable channel chunking specifically
        enable_conv_optimization=True,  # Enable conv optimization
        channel_chunk_size=256,         # Channel chunking size
        batch_chunk_size=32,            # Batch chunking size 
        
    ):
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )
        
        # chunking parameters
        self.enable_chunking = enable_chunking
        self.enable_batch_chunking = enable_batch_chunking and enable_chunking
        self.enable_channel_chunking = enable_channel_chunking and enable_chunking
        self.channel_chunk_size = channel_chunk_size
        self.batch_chunk_size = batch_chunk_size
        self.enable_conv_optimization = enable_conv_optimization
        
        self._tau_buffer(self.beta, learn_beta, channels)
        
    @property
    def beta(self):
        return (self.tau - 1) / self.tau

    # @profile(skip=False, stdout=False, filename="baseline.prof")
    def forward(self, input_):
        self.mem = self._base_state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.output:
            self.spk = self.fire(self.mem) * self.graded_spikes_factor
            return self.spk, self.mem

        else:
            return self.mem
    
    def _tau_buffer(self, beta, learn_beta, channels):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)

        if (
            beta.shape != (channels,)
            and beta.shape != ()
            and beta.shape != (1,)
        ):
            raise ValueError(
                f"Beta shape {beta.shape} must be either ({channels},) or (1,)"
            )

        tau = 1 / (1 - beta + 1e-12)
        if learn_beta:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)
            
    def full_mode_conv1d_truncated(self, input_tensor, kernel_tensor):
        # get dimensions
        batch_size, in_channels, num_steps = input_tensor.shape
        # kernel_tensor: (channels, 1, kernel_size)
        out_channels, _, kernel_size = kernel_tensor.shape

        kernel_tensor = torch.flip(kernel_tensor, dims=[-1]).to(
            input_tensor.device
        )

        # pad the input tensor on both sides
        padding = kernel_size - 1
        padded_input = F.pad(input_tensor, (padding, padding))

        # perform convolution with the padded input
        conv_result = F.conv1d(padded_input, kernel_tensor, groups=in_channels)

        # truncate the result to match the original input length
        # TODO: potential optimization?
        truncated_result = conv_result[..., 0:num_steps]

        return truncated_result

    def _base_state_function(self, input_):
        """Base state function with optional chunking strategies."""
        num_steps, batch, channels = input_.shape
        
        if self.enable_chunking:
            # Use chunking optimization
            decay_filter = self._create_decay_filter(num_steps, channels, input_.device)
            
            # prepare for convolution
            input_ = input_.permute(1, 2, 0)
            decay_filter = decay_filter.permute(1, 0).unsqueeze(1)

            # Choose convolution strategy based on enabled chunking options
            # Priority: batch -> channel -> optimized conv
            if self.enable_batch_chunking and batch > self.batch_chunk_size:
                conv_result = self._chunked_batch_conv(input_, decay_filter)
            elif self.enable_channel_chunking and channels > self.channel_chunk_size:
                conv_result = self._chunked_channel_conv(input_, decay_filter)
            else:
                conv_result = self._optimized_conv1d(input_, decay_filter)

            return conv_result.permute(2, 0, 1)  # return membrane potential trace
        else:
            # Use original StateLeaky implementation
            # make the decay filter
            time_steps = torch.arange(0, num_steps).to(input_.device)
            assert time_steps.shape == (num_steps,)

            # single channel case
            if self.tau.shape == ():
                decay_filter = (
                    torch.exp(-time_steps / self.tau.to(input_.device))
                    .unsqueeze(1)
                    .expand(num_steps, channels)
                )
                assert decay_filter.shape == (num_steps, channels)
            # multichannel case
            else:
                # expand timesteps to be fo shape (num_steps, channels)
                time_steps = time_steps.unsqueeze(1).expand(num_steps, channels)
                # expand tau to be of shape (num_steps, channels)
                tau = (
                    self.tau.unsqueeze(0)
                    .expand(num_steps, channels)
                    .to(input_.device)
                )
                # compute decay filter
                decay_filter = torch.exp(-time_steps / tau)
                assert decay_filter.shape == (num_steps, channels)

            # prepare for convolution
            input_ = input_.permute(1, 2, 0)
            assert input_.shape == (batch, channels, num_steps)
            decay_filter = decay_filter.permute(1, 0).unsqueeze(1)
            assert decay_filter.shape == (channels, 1, num_steps)

            conv_result = self.full_mode_conv1d_truncated(input_, decay_filter)
            assert conv_result.shape == (batch, channels, num_steps)

            return conv_result.permute(2, 0, 1)  # return membrane potential trace

    def _create_decay_filter(self, num_steps, channels, device):
        """Create decay filter with optional pre-flip optimization."""
        if self.enable_conv_optimization:
            # Create time steps in reverse order to eliminate flip operation later
            time_steps = torch.arange(num_steps - 1, -1, -1, dtype=torch.float32, device=device)
        else:
            time_steps = torch.arange(0, num_steps, dtype=torch.float32, device=device)

        # single channel case
        if self.tau.shape == ():
            decay_filter = torch.exp(-time_steps / self.tau.to(device)).unsqueeze(1).expand(num_steps, channels)
        # multichannel case
        else:
            time_steps = time_steps.unsqueeze(1).expand(num_steps, channels)
            tau = self.tau.unsqueeze(0).expand(num_steps, channels).to(device)
            decay_filter = torch.exp(-time_steps / tau)

        return decay_filter

    def _chunked_batch_conv(self, input_tensor, decay_filter):
        """Process batches in chunks to reduce peak memory usage - V4 addition."""
        total_batch, channels, num_steps = input_tensor.shape
        chunk_size = min(self.batch_chunk_size, total_batch)
        
        results = []
        for start_batch in range(0, total_batch, chunk_size):
            end_batch = min(start_batch + chunk_size, total_batch)
            
            # Process this batch chunk
            batch_input = input_tensor[start_batch:end_batch, :, :]
            
            # Process chunk with channel chunking if enabled and needed
            if self.enable_channel_chunking and channels > self.channel_chunk_size:
                batch_result = self._chunked_channel_conv(batch_input, decay_filter)
            else:
                batch_result = self._optimized_conv1d(batch_input, decay_filter)
            
            results.append(batch_result)
        
        return torch.cat(results, dim=0)

    def _chunked_channel_conv(self, input_tensor, decay_filter):
        """Process channels in chunks to reduce peak memory usage."""
        batch, channels, num_steps = input_tensor.shape
        chunk_size = min(self.channel_chunk_size, channels)
        
        results = []
        for start_ch in range(0, channels, chunk_size):
            end_ch = min(start_ch + chunk_size, channels)
            
            # Process this channel chunk
            input_chunk = input_tensor[:, start_ch:end_ch, :]
            filter_chunk = decay_filter[start_ch:end_ch, :, :]
            
            chunk_result = self._optimized_conv1d(input_chunk, filter_chunk)
            results.append(chunk_result)
        
        return torch.cat(results, dim=1)
    

    def _optimized_conv1d(self, input_tensor, kernel_tensor):
        """Optimized convolution with optional pre-flipped kernel."""
        # Apply flip only if conv optimization is disabled
        if not self.enable_conv_optimization:
            kernel_tensor = torch.flip(kernel_tensor, dims=[-1])
        
        kernel_tensor = kernel_tensor.to(input_tensor.device)

        # pad the input tensor on both sides
        padding = kernel_tensor.shape[-1] - 1
        padded_input = F.pad(input_tensor, (padding, padding))

        # perform convolution with the padded input
        conv_result = F.conv1d(padded_input, kernel_tensor, groups=input_tensor.shape[1])

        # truncate the result to match the original input length
        truncated_result = conv_result[..., :input_tensor.shape[-1]]

        return truncated_result

    def get_memory_info(self, input_shape):
        """Estimate memory usage for given input shape - V4 with batch chunking."""
        num_steps, batch, channels = input_shape
        bytes_per_float = 4  # float32
        
        # Original memory usage
        original_memory = num_steps * channels * batch * bytes_per_float / 1e6
        
        # With batch chunking only
        effective_batch = min(self.batch_chunk_size, batch)
        batch_chunked_memory = num_steps * channels * effective_batch * bytes_per_float / 1e6
        
        # With channel chunking only
        effective_channels = min(self.channel_chunk_size, channels)
        channel_chunked_memory = num_steps * effective_channels * batch * bytes_per_float / 1e6
        
        # With batch and channel chunking
        batch_channel_chunked_memory = num_steps * effective_channels * effective_batch * bytes_per_float / 1e6
        

        
        return {
            "original_mb": original_memory,
            "batch_chunked_mb": batch_chunked_memory,
            "channel_chunked_mb": channel_chunked_memory,
            "batch_channel_chunked_mb": batch_channel_chunked_memory,
       
            "memory_reduction_factor": original_memory / batch_chunked_memory if batch_chunked_memory > 0 else float('inf')
        }


def test_all_optimizations():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing StateLeaky_chunking on {device}")
    
    # Test parameters 
    timesteps = 1000
    batch = 128  #
    channels = 2054
    
    input_tensor = torch.randn(timesteps, batch, channels).to(device)
    print(f"Input shape: {input_tensor.shape}")
    
    # Create models - demonstrating different chunking strategies
    original = StateLeaky_chunking(beta=0.9, channels=channels, enable_chunking=False).to(device)
    
    # Only batch chunking
    batch_only = StateLeaky_chunking(
        beta=0.9, channels=channels, 
        enable_batch_chunking=True, enable_channel_chunking=False,
        batch_chunk_size=32
    ).to(device)
    
    # Only channel chunking  
    channel_only = StateLeaky_chunking(
        beta=0.9, channels=channels,
        enable_batch_chunking=False, enable_channel_chunking=True,
        channel_chunk_size=128
    ).to(device)
    
    # Both batch and channel chunking
    both_chunking = StateLeaky_chunking(
        beta=0.9, channels=channels,
        enable_batch_chunking=True, enable_channel_chunking=True,
        batch_chunk_size=32, channel_chunk_size=128
    ).to(device)
    
    # Default chunking (both enabled with default sizes)
    default_chunking = StateLeaky_chunking(beta=0.9, channels=channels).to(device)

    
    # Memory info
    memory_info = default_chunking.get_memory_info(input_tensor.shape)
    print(f"Memory reduction potential: {memory_info['memory_reduction_factor']:.1f}x")
    print(f"Original memory: {memory_info['original_mb']:.1f} MB")
    print(f"Batch chunked: {memory_info['batch_chunked_mb']:.1f} MB")
    print(f"Channel chunked: {memory_info['channel_chunked_mb']:.1f} MB")
    print(f"Batch + Channel chunked: {memory_info['batch_channel_chunked_mb']:.1f} MB")

    
    # Benchmark performance
    models = [
        ("Original (no chunking)", original),
        ("Batch chunking only", batch_only),
        ("Channel chunking only", channel_only), 
        ("Both batch+channel", both_chunking),
        ("Default (both enabled)", default_chunking)
    ]
    
    results = {}
    
    with torch.no_grad():
        for name, model in models:
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.time()
            output = model(input_tensor)
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.time() - start
            
            results[name] = (output, elapsed)
            print(f"{name}: {elapsed*1000:.2f}ms, shape: {output[0].shape}")
    
    # Accuracy comparison
    orig_output = results["Original (no chunking)"][0][0]
    print(f"Accuracy comparison (vs Original):")
    
    for name, (output, _) in results.items():
        if name != "Original (no chunking)":
            diff = torch.abs(orig_output - output[0]).mean()
            print(f"{name}: {diff:.6f} mean difference")
    
    # Performance summary
    orig_time = results["Original (no chunking)"][1]
    print(f"Speedup comparison (vs Original):")
    for name, (_, elapsed) in results.items():
        if name != "Original (no chunking)":
            speedup = orig_time / elapsed
            print(f"{name}: {speedup:.2f}x speedup")
    
    print(f"RESULTS: Memory reduction up to {memory_info['memory_reduction_factor']:.1f}x with StateLeaky_chunking batch chunking!")


# TODO: throw exceptions if calling subclass methods we don't want to use
# fire_inhibition
# mem_reset, init, detach, zeros, reset_mem, init_leaky
# detach_hidden, reset_hidden


if __name__ == "__main__":
    # test
    device = "cuda"
    print("=== StateLeaky_chunking Optimization Test ===")
    test_all_optimizations()
