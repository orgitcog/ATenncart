# Tensor Logic Integration Guide

This guide helps developers integrate Tensor Logic functionality into their OpenCart applications.

## Quick Start

### 1. Basic Setup

The Tensor Logic extension is automatically available once installed. No additional configuration is required.

### 2. Using Tensors in Your Code

```php
<?php
namespace Opencart\Catalog\Controller\Extension\MyExtension;

use Opencart\System\Library\Tensor\Tensor;

class MyController extends \Opencart\System\Engine\Controller {
    public function index(): void {
        // Create a tensor
        $data = [[1, 2, 3], [4, 5, 6]];
        $tensor = Tensor::create($data);
        
        // Perform operations
        $doubled = $tensor->mul(2);
        $sum = $tensor->sum();
        
        // Use results
        $this->response->setOutput(json_encode([
            'original' => $tensor->toArray(),
            'doubled' => $doubled->toArray(),
            'sum' => $sum
        ]));
    }
}
```

### 3. Using Neural Networks

```php
<?php
use Opencart\System\Library\Nn\Model;
use Opencart\System\Library\Nn\Dense;
use Opencart\System\Library\Tensor\Tensor;

// Load a pre-trained model (from database or file)
$modelJson = $this->config->get('my_model_json');
$model = Model::load($modelJson);

// Make predictions
$input = Tensor::create([1.0, 2.0, 3.0]);
$output = $model->predict($input);
$prediction = $output->toArray();
```

## Common Use Cases

### Product Recommendations

```php
<?php
// Calculate user-product interaction matrix
$userRatings = Tensor::create($userProductMatrix);

// Find similar users by dot product
$userVector = Tensor::create($currentUserRatings);
$similarities = [];

foreach ($otherUsers as $userId => $ratings) {
    $otherVector = Tensor::create($ratings);
    $similarity = $userVector->mul($otherVector)->sum();
    $similarities[$userId] = $similarity;
}

// Get recommendations from most similar users
arsort($similarities);
$recommendedProducts = $this->getRecommendationsFromUsers(array_keys($similarities));
```

### Dynamic Pricing

```php
<?php
// Features: [demand, competition, seasonality]
$features = [
    $this->getDemandScore($productId),
    $this->getCompetitionScore($productId),
    $this->getSeasonalityScore()
];

$input = Tensor::create($features);
$priceMultiplier = $pricingModel->predict($input)->getData()[0];

$finalPrice = $basePrice * $priceMultiplier;
```

### Customer Segmentation

```php
<?php
// Calculate customer feature vectors
$customerFeatures = [
    'purchases_per_month' => $this->getAveragePurchases($customerId),
    'average_order_value' => $this->getAverageOrderValue($customerId),
    'engagement_score' => $this->getEngagementScore($customerId)
];

$customerVector = Tensor::create(array_values($customerFeatures));

// Find similar customers
$allCustomers = $this->model_customer_customer->getCustomers();
$similarities = [];

foreach ($allCustomers as $customer) {
    $otherVector = Tensor::create($this->getCustomerFeatures($customer['customer_id']));
    $similarity = $customerVector->mul($otherVector)->sum();
    $similarities[$customer['customer_id']] = $similarity;
}
```

## API Integration

### Calling from JavaScript

```javascript
// Create tensor via API
fetch('/index.php?route=extension/tensor_logic/api/tensor_logic.create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        data: [[1, 2], [3, 4]]
    })
})
.then(response => response.json())
.then(data => console.log(data));

// Matrix multiplication
fetch('/index.php?route=extension/tensor_logic/api/tensor_logic.matmul', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        a: [[1, 2], [3, 4]],
        b: [[5, 6], [7, 8]]
    })
})
.then(response => response.json())
.then(data => console.log('Result:', data.result.data));
```

### Python Integration

```python
import requests
import json

# Create model
response = requests.get('http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.createModel',
    params={'input_size': 4, 'hidden_size': 8, 'output_size': 2})
model = response.json()['model']

# Make prediction
response = requests.post('http://localhost/index.php?route=extension/tensor_logic/api/tensor_logic.predict',
    json={
        'model': model,
        'input': [1.0, 2.0, 3.0, 4.0]
    })
prediction = response.json()['output']['data']
print('Prediction:', prediction)
```

## Performance Tips

### 1. Batch Operations

Process multiple items at once using batched tensors:

```php
// Instead of this:
foreach ($products as $product) {
    $tensor = Tensor::create($product['features']);
    $predictions[] = $model->predict($tensor);
}

// Do this:
$batchFeatures = array_map(fn($p) => $p['features'], $products);
$batchTensor = Tensor::create($batchFeatures);
$predictions = $model->predict($batchTensor)->toArray();
```

### 2. Cache Models

Load models once and cache them:

```php
if (!isset($this->cache->get('pricing_model'))) {
    $modelJson = file_get_contents('path/to/model.json');
    $this->cache->set('pricing_model', $modelJson);
}

$model = Model::load($this->cache->get('pricing_model'));
```

### 3. Pre-compute When Possible

For static data, pre-compute tensor operations:

```php
// Pre-compute product similarity matrix
$productFeatures = $this->model_catalog_product->getAllProductFeatures();
$similarityMatrix = [];

foreach ($productFeatures as $i => $features1) {
    $tensor1 = Tensor::create($features1);
    foreach ($productFeatures as $j => $features2) {
        if ($i < $j) {
            $tensor2 = Tensor::create($features2);
            $similarity = $tensor1->mul($tensor2)->sum();
            $similarityMatrix[$i][$j] = $similarity;
        }
    }
}

// Store in database or cache
$this->cache->set('product_similarities', $similarityMatrix, 86400);
```

## Model Training

**Note:** This extension is designed for inference only. Train your models externally using:

- **PyTorch**: Train in Python, export to JSON
- **TensorFlow**: Train in Python, convert weights to JSON
- **Custom**: Any ML framework that can export to JSON format

### Example: Export from PyTorch

```python
import torch
import json

# Train your model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2),
    torch.nn.Softmax(dim=1)
)

# Train model here...

# Export to OpenCart format
opencart_model = {
    'version': '1.0',
    'layers': []
}

for i, layer in enumerate(model):
    if isinstance(layer, torch.nn.Linear):
        opencart_model['layers'].append({
            'type': 'dense',
            'input_size': layer.in_features,
            'output_size': layer.out_features,
            'weights': {
                'shape': [layer.in_features, layer.out_features],
                'data': layer.weight.T.detach().numpy().flatten().tolist()
            },
            'bias': {
                'shape': [layer.out_features],
                'data': layer.bias.detach().numpy().tolist()
            },
            'activation': 'relu' if i < len(model) - 1 else 'softmax'
        })

with open('model.json', 'w') as f:
    json.dump(opencart_model, f, indent=2)
```

## Troubleshooting

### Issue: "Class not found" errors

**Solution:** Ensure the library files are in the correct location:
- `upload/system/library/tensor/tensor.php`
- `upload/system/library/nn/*.php`

### Issue: Memory issues with large tensors

**Solution:** PHP may have memory limits. For large operations:
1. Increase `memory_limit` in php.ini
2. Process data in batches
3. Consider external API for heavy computations

### Issue: Slow performance

**Solution:**
1. Enable OpCache in PHP
2. Cache computed results
3. Use batched operations
4. Consider moving heavy operations to a background queue

## Best Practices

1. **Validate Input**: Always validate tensor shapes match expectations
2. **Error Handling**: Wrap tensor operations in try-catch blocks
3. **Document Models**: Keep track of model architecture and version
4. **Test Predictions**: Verify output ranges and shapes
5. **Monitor Performance**: Log operation times for optimization

## Support & Resources

- **Documentation**: `upload/extension/tensor-logic/README.md`
- **Examples**: `test_tensor_logic.php`, `examples_tensor_logic.php`
- **API Reference**: `upload/extension/tensor-logic/index.html`
- **Website**: https://tensor-logic.org

## License

GPL-3.0-or-later (same as OpenCart)
