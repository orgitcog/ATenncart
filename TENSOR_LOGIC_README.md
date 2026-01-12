# ATenncart - OpenCart with Tensor Logic

OpenCart e-commerce platform enhanced with tensor operations and neural network capabilities, implementing principles from **tensor-logic.org** with **ATen** (A Tensor Library) and **nn** (Neural Networks).

## 🎯 What's New

This fork of OpenCart includes **Tensor Logic** - a machine learning inference system that brings PyTorch-inspired tensor operations and neural networks to PHP-based e-commerce.

### Key Features

- 🔢 **ATen-inspired Tensor Operations**: Multi-dimensional arrays with mathematical operations
- 🧠 **Neural Network Inference**: Forward propagation with dense layers and activations
- 🔌 **REST API**: JSON-based endpoints for tensor operations and ML inference
- 📊 **E-commerce Ready**: Examples for recommendations, pricing, and forecasting

## 🚀 Quick Start

### Installation

1. Clone this repository
2. Follow standard [OpenCart installation](INSTALL.md)
3. The Tensor Logic extension is included and ready to use

### Basic Usage

#### Via PHP

```php
use Opencart\System\Library\Tensor\Tensor;
use Opencart\System\Library\Nn\Model;
use Opencart\System\Library\Nn\Dense;

// Create tensors
$a = Tensor::create([[1, 2], [3, 4]]);
$b = Tensor::create([[5, 6], [7, 8]]);

// Perform operations
$sum = $a->add($b);
$product = $a->matmul($b);

// Create neural network
$model = new Model();
$model->addDense(new Dense(3, 5), 'relu');
$model->addDense(new Dense(5, 2), 'softmax');

// Make predictions
$input = Tensor::create([1.0, 2.0, 3.0]);
$output = $model->predict($input);
```

#### Via REST API

```bash
# Create tensor
curl -X POST http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.create \
  -H "Content-Type: application/json" \
  -d '{"data": [[1, 2], [3, 4]]}'

# Matrix multiplication
curl -X POST http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.matmul \
  -H "Content-Type: application/json" \
  -d '{"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]}'

# Neural network inference
curl -X POST http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.predict \
  -H "Content-Type: application/json" \
  -d '{"model": {...}, "input": [1.0, 2.0, 3.0]}'
```

## 📚 Documentation

- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Overview of what was built
- **[Integration Guide](TENSOR_LOGIC_INTEGRATION.md)** - How to use in your code
- **[Extension README](upload/extension/tensor-logic/README.md)** - API reference and details
- **[HTML Docs](upload/extension/tensor-logic/index.html)** - Interactive documentation

## 🧪 Examples & Tests

### Run Tests

```bash
php test_tensor_logic.php
```

This runs 10 unit tests covering:
- Tensor creation and operations
- Matrix multiplication
- Activation functions
- Neural network models
- Serialization

### Run E-commerce Examples

```bash
php examples_tensor_logic.php
```

This demonstrates 5 real-world use cases:
1. **Product Recommendations** - Collaborative filtering with tensors
2. **Dynamic Pricing** - Neural network-based price optimization
3. **Customer Segmentation** - Feature analysis and similarity
4. **Inventory Optimization** - Multi-warehouse tensor computations
5. **Sales Prediction** - Neural network forecasting

## 🏗️ Architecture

```
upload/
├── extension/
│   └── tensor-logic/           # Extension files
│       ├── install.json
│       ├── README.md
│       ├── index.html
│       └── catalog/
│           └── controller/
│               └── api/
│                   └── tensor_logic.php
└── system/
    └── library/
        ├── tensor/             # ATen-inspired operations
        │   └── tensor.php
        └── nn/                 # Neural network components
            ├── activation.php
            ├── dense.php
            └── model.php
```

## 💡 Use Cases

### Product Recommendations

```php
// Calculate user-product similarity
$userRatings = Tensor::create($userProductMatrix);
$similarities = $userRatings->matmul($userRatings->transpose());
```

### Dynamic Pricing

```php
// Use neural network for price optimization
$features = Tensor::create([$demand, $competition, $seasonality]);
$priceMultiplier = $pricingModel->predict($features)->getData()[0];
```

### Customer Segmentation

```php
// Find similar customers
$customerFeatures = Tensor::create([$purchases, $avgOrder, $engagement]);
$similarity = $customerFeatures->mul($otherCustomerFeatures)->sum();
```

## 🔧 Requirements

- PHP 8.0 or higher
- OpenCart 4.x
- JSON extension (standard in PHP)

## ✅ Quality Assurance

All code passes:
- ✅ PHPStan Level 6 static analysis
- ✅ php-cs-fixer code style checks
- ✅ PHP syntax validation
- ✅ 10/10 unit tests
- ✅ 5/5 example scenarios

## 🤝 Contributing

This is a demonstration of integrating tensor operations with OpenCart. For:
- **OpenCart core**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Tensor Logic improvements**: Open issues or PRs on this repository

## 📄 License

GPL-3.0-or-later (same as OpenCart)

See [LICENSE.md](LICENSE.md)

## 🌟 Credits

**Tensor Logic Implementation:**
- Inspired by PyTorch ATen: https://pytorch.org/cppdocs/
- Inspired by PyTorch nn: https://pytorch.org/docs/stable/nn.html
- Based on tensor-logic.org principles

**OpenCart Platform:**
- Original OpenCart: https://www.opencart.com
- GitHub: https://github.com/opencart/opencart

## 🔗 Links

- **OpenCart Homepage**: https://www.opencart.com/
- **OpenCart Forums**: https://forum.opencart.com/
- **OpenCart Docs**: http://docs.opencart.com/
- **Tensor Logic**: https://tensor-logic.org

## 🎓 Learning Resources

- [Test file](test_tensor_logic.php) - Basic operations and usage
- [Examples file](examples_tensor_logic.php) - Real-world scenarios
- [Integration guide](TENSOR_LOGIC_INTEGRATION.md) - Developer reference
- [API documentation](upload/extension/tensor-logic/index.html) - Endpoint reference

---

**Ready to bring machine learning to your e-commerce platform!** 🚀
