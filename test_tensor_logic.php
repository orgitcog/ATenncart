<?php
/**
 * Test script for Tensor Logic implementation
 * 
 * Run from command line: php test_tensor_logic.php
 */

// Include required classes
require_once __DIR__ . '/upload/system/library/tensor/tensor.php';
require_once __DIR__ . '/upload/system/library/nn/activation.php';
require_once __DIR__ . '/upload/system/library/nn/dense.php';
require_once __DIR__ . '/upload/system/library/nn/model.php';

use Opencart\System\Library\Tensor\Tensor;
use Opencart\System\Library\Nn\Model;
use Opencart\System\Library\Nn\Dense;
use Opencart\System\Library\Nn\Activation;

echo "=== Tensor Logic Test Suite ===\n\n";

// Test 1: Basic tensor creation
echo "Test 1: Tensor Creation\n";
$tensor1 = Tensor::create([[1, 2], [3, 4]]);
echo "Created tensor with shape: [" . implode(", ", $tensor1->getShape()) . "]\n";
echo "Data: " . json_encode($tensor1->toArray()) . "\n\n";

// Test 2: Tensor operations
echo "Test 2: Tensor Addition\n";
$a = Tensor::create([[1, 2], [3, 4]]);
$b = Tensor::create([[5, 6], [7, 8]]);
$sum = $a->add($b);
echo "A + B = " . json_encode($sum->toArray()) . "\n\n";

// Test 3: Matrix multiplication
echo "Test 3: Matrix Multiplication\n";
$result = $a->matmul($b);
echo "A @ B = " . json_encode($result->toArray()) . "\n\n";

// Test 4: Tensor factory methods
echo "Test 4: Factory Methods\n";
$zeros = Tensor::zeros([2, 2]);
$ones = Tensor::ones([2, 2]);
echo "Zeros: " . json_encode($zeros->toArray()) . "\n";
echo "Ones: " . json_encode($ones->toArray()) . "\n\n";

// Test 5: Activation functions
echo "Test 5: Activation Functions\n";
$input = Tensor::create([-1, 0, 1, 2]);
$relu = Activation::relu($input);
$sigmoid = Activation::sigmoid($input);
echo "Input: " . json_encode($input->toArray()) . "\n";
echo "ReLU: " . json_encode($relu->toArray()) . "\n";
echo "Sigmoid: " . json_encode($sigmoid->toArray()) . "\n\n";

// Test 6: Neural network model
echo "Test 6: Neural Network Model\n";
$model = new Model();
$model->addDense(new Dense(3, 5), 'relu');
$model->addDense(new Dense(5, 2), 'softmax');

echo $model->summary() . "\n";

// Test 7: Neural network inference
echo "Test 7: Neural Network Inference\n";
$input = Tensor::create([1.0, 2.0, 3.0]);
$output = $model->predict($input);
echo "Input: " . json_encode($input->toArray()) . "\n";
echo "Output: " . json_encode($output->toArray()) . "\n";
echo "Output sum (should be ~1.0 for softmax): " . $output->sum() . "\n\n";

// Test 8: Model serialization
echo "Test 8: Model Serialization\n";
$modelJson = $model->save();
echo "Model saved to JSON (" . strlen($modelJson) . " bytes)\n";
$loadedModel = Model::load($modelJson);
echo "Model loaded successfully\n";
$output2 = $loadedModel->predict($input);
echo "Output after reload: " . json_encode($output2->toArray()) . "\n\n";

// Test 9: Tensor serialization
echo "Test 9: Tensor Serialization\n";
$original = Tensor::create([[1, 2, 3], [4, 5, 6]]);
$json = $original->toJson();
$restored = Tensor::fromJson($json);
echo "Original: " . json_encode($original->toArray()) . "\n";
echo "Restored: " . json_encode($restored->toArray()) . "\n\n";

// Test 10: Reductions
echo "Test 10: Reduction Operations\n";
$tensor = Tensor::create([[1, 2, 3], [4, 5, 6]]);
echo "Tensor: " . json_encode($tensor->toArray()) . "\n";
echo "Sum: " . $tensor->sum() . "\n";
echo "Mean: " . $tensor->mean() . "\n";
echo "Max: " . $tensor->max() . "\n";
echo "Min: " . $tensor->min() . "\n\n";

echo "=== All tests completed successfully! ===\n";
