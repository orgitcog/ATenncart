# Tensor Logic Extension for OpenCart

Implementation of tensor-logic.org principles with ATen & nn inspired operations for OpenCart.

## Overview

This extension brings tensor operations and neural network capabilities to OpenCart, inspired by PyTorch's ATen (A Tensor Library) and nn (Neural Network) modules. It provides a PHP-based implementation that allows e-commerce applications to leverage tensor computations and neural network inference.

## Features

### ATen-inspired Tensor Operations
- **Tensor creation**: Create tensors from nested arrays
- **Factory methods**: `zeros()`, `ones()`, `rand()` for tensor initialization
- **Element-wise operations**: `add()`, `sub()`, `mul()`, `div()`
- **Matrix operations**: `matmul()` (matrix multiplication), `transpose()`
- **Reduction operations**: `sum()`, `mean()`, `max()`, `min()`
- **Serialization**: `toJson()`, `fromJson()` for persistence

### nn-inspired Neural Network Layers
- **Dense (Fully Connected) layers**: Configurable input/output sizes with Xavier initialization
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
- **Sequential Model**: Chain layers together for inference
- **Model serialization**: Save and load models as JSON

## Installation

1. Copy the `tensor-logic` extension to `upload/extension/`
2. Copy tensor and nn libraries to `upload/system/library/`
3. Access the extension through OpenCart admin panel

## API Endpoints

### Base URL
All endpoints are accessed via: `/index.php?route=extension/tensor_logic/api/tensor_logic`

### GET / - API Information
```bash
curl http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic
```

Returns API version, available endpoints, and documentation links.

### POST /create - Create Tensor
```bash
curl -X POST http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.create \
  -H "Content-Type: application/json" \
  -d '{"data": [[1, 2], [3, 4]]}'
```

### POST /add - Add Tensors
```bash
curl -X POST http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.add \
  -H "Content-Type: application/json" \
  -d '{"a": [[1, 2]], "b": [[3, 4]]}'
```

### POST /multiply - Element-wise Multiplication
```bash
curl -X POST http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.multiply \
  -H "Content-Type: application/json" \
  -d '{"a": [[1, 2]], "b": [[3, 4]]}'
```

### POST /matmul - Matrix Multiplication
```bash
curl -X POST http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.matmul \
  -H "Content-Type: application/json" \
  -d '{"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]}'
```

### GET /createModel - Create Example Neural Network
```bash
curl "http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.createModel?input_size=3&hidden_size=5&output_size=2"
```

### POST /predict - Neural Network Inference
```bash
curl -X POST http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": { ... },
    "input": [1.0, 2.0, 3.0]
  }'
```

## PHP Usage Examples

### Tensor Operations

```php
use Opencart\System\Library\Tensor\Tensor;

// Create tensors
$a = Tensor::create([[1, 2], [3, 4]]);
$b = Tensor::create([[5, 6], [7, 8]]);

// Basic operations
$sum = $a->add($b);
$product = $a->mul($b);
$matmul = $a->matmul($b);

// Reductions
$total = $a->sum();
$average = $a->mean();
$maximum = $a->max();

// Factory methods
$zeros = Tensor::zeros([3, 3]);
$ones = Tensor::ones([2, 4]);
$random = Tensor::rand([5, 5]);

// Serialization
$json = $a->toJson();
$restored = Tensor::fromJson($json);
```

### Neural Network

```php
use Opencart\System\Library\Nn\Model;
use Opencart\System\Library\Nn\Dense;
use Opencart\System\Library\Tensor\Tensor;

// Create a simple neural network
$model = new Model();
$model->addDense(new Dense(3, 5), 'relu');
$model->addDense(new Dense(5, 2), 'softmax');

// Make predictions
$input = Tensor::create([1.0, 2.0, 3.0]);
$output = $model->predict($input);

// Get model summary
echo $model->summary();

// Save model
$modelJson = $model->save();
file_put_contents('model.json', $modelJson);

// Load model
$loadedModel = Model::load(file_get_contents('model.json'));
```

## Architecture

### Directory Structure
```
upload/
├── extension/
│   └── tensor-logic/
│       ├── install.json
│       ├── admin/
│       │   └── controller/
│       └── catalog/
│           └── controller/
│               └── api/
│                   └── tensor_logic.php
└── system/
    └── library/
        ├── tensor/
        │   └── tensor.php
        └── nn/
            ├── activation.php
            ├── dense.php
            └── model.php
```

### Core Components

1. **Tensor Library** (`system/library/tensor/tensor.php`)
   - ATen-inspired tensor operations
   - Multi-dimensional array support
   - Element-wise and matrix operations

2. **Neural Network Library** (`system/library/nn/`)
   - `activation.php`: Activation functions
   - `dense.php`: Fully connected layers
   - `model.php`: Sequential model container

3. **API Controller** (`extension/tensor-logic/catalog/controller/api/tensor_logic.php`)
   - REST API endpoints
   - JSON request/response handling
   - Error handling and validation

## Use Cases

- **Product Recommendations**: Use neural networks for collaborative filtering
- **Price Optimization**: Apply tensor operations for dynamic pricing algorithms
- **Customer Segmentation**: Cluster analysis using tensor computations
- **Demand Forecasting**: Time series prediction with neural networks
- **Image Classification**: Product categorization using pre-trained models

## Requirements

- PHP 8.0 or higher
- OpenCart 4.x
- JSON extension enabled

## Performance Considerations

- This is a PHP implementation and not optimized for large-scale tensor operations
- For production machine learning workloads, consider using external services (e.g., via API)
- Suitable for inference with small to medium-sized models
- Training functionality is not included; models should be trained externally

## Credits

Inspired by:
- PyTorch ATen: https://pytorch.org/cppdocs/
- PyTorch nn module: https://pytorch.org/docs/stable/nn.html
- tensor-logic.org principles

## License

GPL-3.0-or-later (same as OpenCart)

## Support

For issues and questions, visit: https://tensor-logic.org
