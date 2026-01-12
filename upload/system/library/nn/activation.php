<?php
namespace Opencart\System\Library\Nn;

use Opencart\System\Library\Tensor\Tensor;

/**
 * Activation functions for neural networks
 */
class Activation {
	/**
	 * ReLU activation: max(0, x)
	 *
	 * @param Tensor $input
	 *
	 * @return Tensor
	 */
	public static function relu(Tensor $input): Tensor {
		return $input->apply(fn ($x) => max(0, $x));
	}

	/**
	 * Sigmoid activation: 1 / (1 + exp(-x))
	 *
	 * @param Tensor $input
	 *
	 * @return Tensor
	 */
	public static function sigmoid(Tensor $input): Tensor {
		return $input->apply(fn ($x) => 1.0 / (1.0 + exp(-$x)));
	}

	/**
	 * Tanh activation
	 *
	 * @param Tensor $input
	 *
	 * @return Tensor
	 */
	public static function tanh(Tensor $input): Tensor {
		return $input->apply(fn ($x) => tanh($x));
	}

	/**
	 * Softmax activation (for 1D tensor)
	 *
	 * @param Tensor $input
	 *
	 * @return Tensor
	 */
	public static function softmax(Tensor $input): Tensor {
		$data = $input->getData();
		$max = max($data);

		// Numerical stability: subtract max before exp
		$exp = array_map(fn ($x) => exp($x - $max), $data);
		$sum = array_sum($exp);

		$result = array_map(fn ($x) => $x / $sum, $exp);

		return Tensor::create($result);
	}

	/**
	 * LeakyReLU activation
	 *
	 * @param Tensor $input
	 * @param float  $alpha Negative slope coefficient
	 *
	 * @return Tensor
	 */
	public static function leakyRelu(Tensor $input, float $alpha = 0.01): Tensor {
		return $input->apply(fn ($x) => $x > 0 ? $x : $alpha * $x);
	}
}
