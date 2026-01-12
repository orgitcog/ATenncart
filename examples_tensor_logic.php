<?php
/**
 * Tensor Logic - Practical Examples
 * 
 * Demonstrates real-world use cases for tensor-logic in e-commerce
 */

require_once __DIR__ . '/upload/system/library/tensor/tensor.php';
require_once __DIR__ . '/upload/system/library/nn/activation.php';
require_once __DIR__ . '/upload/system/library/nn/dense.php';
require_once __DIR__ . '/upload/system/library/nn/model.php';

use Opencart\System\Library\Tensor\Tensor;
use Opencart\System\Library\Nn\Model;
use Opencart\System\Library\Nn\Dense;

echo "=== Tensor Logic - E-commerce Examples ===\n\n";

// Example 1: Product Recommendation System
echo "Example 1: Product Recommendation (Collaborative Filtering)\n";
echo str_repeat("-", 60) . "\n";

// User-item interaction matrix (5 users, 4 products)
// Values represent ratings (0-5)
$userItemMatrix = [
    [5, 3, 0, 1],  // User 1
    [4, 0, 0, 1],  // User 2
    [1, 1, 0, 5],  // User 3
    [1, 0, 0, 4],  // User 4
    [0, 1, 5, 4],  // User 5
];

$ratings = Tensor::create($userItemMatrix);
echo "User-Item Rating Matrix:\n";
print_r($ratings->toArray());

// Calculate mean rating per user
echo "\nUser statistics:\n";
for ($i = 0; $i < 5; $i++) {
    $userRatings = Tensor::create($userItemMatrix[$i]);
    echo "User " . ($i + 1) . " - Average rating: " . 
         number_format($userRatings->sum() / 4, 2) . "\n";
}

echo "\n";

// Example 2: Dynamic Pricing Model
echo "Example 2: Dynamic Pricing with Neural Network\n";
echo str_repeat("-", 60) . "\n";

// Create a pricing model: [demand, competition, season] -> price_multiplier
$pricingModel = new Model();
$pricingModel->addDense(new Dense(3, 8), 'relu');
$pricingModel->addDense(new Dense(8, 4), 'relu');
$pricingModel->addDense(new Dense(4, 1), 'sigmoid');

echo "Pricing Model Architecture:\n";
echo $pricingModel->summary();

// Test scenarios
$scenarios = [
    'High demand, low competition, peak season' => [0.9, 0.3, 1.0],
    'Low demand, high competition, off-season' => [0.2, 0.8, 0.1],
    'Medium demand, medium competition, normal' => [0.5, 0.5, 0.5],
];

echo "\nPricing predictions (0-1 multiplier):\n";
foreach ($scenarios as $scenario => $features) {
    $input = Tensor::create($features);
    $output = $pricingModel->predict($input);
    $multiplier = $output->getData()[0];
    echo "  $scenario\n";
    echo "    Multiplier: " . number_format($multiplier, 3) . 
         " (Price: $100 → $" . number_format(100 * $multiplier, 2) . ")\n";
}

echo "\n";

// Example 3: Customer Segmentation (Simplified)
echo "Example 3: Customer Feature Analysis\n";
echo str_repeat("-", 60) . "\n";

// Customer features: [purchase_frequency, avg_order_value, engagement_score]
$customers = [
    'Customer A' => [10, 150, 0.8],
    'Customer B' => [2, 50, 0.3],
    'Customer C' => [25, 300, 0.9],
    'Customer D' => [5, 100, 0.5],
];

echo "Customer feature vectors:\n";
$customerTensors = [];
foreach ($customers as $name => $features) {
    $tensor = Tensor::create($features);
    $customerTensors[$name] = $tensor;
    echo "$name: [" . implode(", ", $features) . 
         "] -> Score: " . number_format($tensor->sum(), 2) . "\n";
}

// Calculate similarity (simplified cosine similarity using dot product)
echo "\nCustomer similarity matrix:\n";
$names = array_keys($customers);
for ($i = 0; $i < count($names); $i++) {
    for ($j = $i + 1; $j < count($names); $j++) {
        $t1 = $customerTensors[$names[$i]];
        $t2 = $customerTensors[$names[$j]];
        
        // Dot product as similarity measure
        $similarity = $t1->mul($t2)->sum();
        echo "  {$names[$i]} <-> {$names[$j]}: " . 
             number_format($similarity, 2) . "\n";
    }
}

echo "\n";

// Example 4: Inventory Optimization
echo "Example 4: Multi-Product Inventory Tensor Operations\n";
echo str_repeat("-", 60) . "\n";

// Product inventory: [warehouse1, warehouse2, warehouse3]
$inventory = Tensor::create([
    [100, 150, 200],  // Product A
    [50, 75, 100],    // Product B
    [200, 250, 300],  // Product C
]);

echo "Current inventory levels:\n";
print_r($inventory->toArray());

// Demand forecast (as multipliers)
$demandForecast = Tensor::create([
    [1.2, 1.5, 1.1],  // Expected demand multipliers
    [0.8, 1.0, 0.9],
    [1.5, 1.3, 1.4],
]);

echo "\nDemand forecast multipliers:\n";
print_r($demandForecast->toArray());

// Calculate required stock levels
$requiredStock = $inventory->mul($demandForecast);
echo "\nRequired stock levels:\n";
print_r($requiredStock->toArray());

// Calculate total inventory value per product (assuming price vector)
$prices = Tensor::create([10, 25, 15]);  // Price per unit
echo "\nInventory value per product:\n";
$inventoryByProduct = [
    $inventory->toArray()[0],
    $inventory->toArray()[1],
    $inventory->toArray()[2],
];
$priceData = $prices->toArray();
for ($i = 0; $i < 3; $i++) {
    $productInventory = Tensor::create($inventoryByProduct[$i]);
    $totalUnits = $productInventory->sum();
    $value = $totalUnits * $priceData[$i];
    echo "  Product " . chr(65 + $i) . ": " . 
         number_format($totalUnits, 0) . " units × $" . 
         number_format($priceData[$i], 2) . " = $" . 
         number_format($value, 2) . "\n";
}

echo "\n";

// Example 5: Sales Prediction Model
echo "Example 5: Sales Prediction Neural Network\n";
echo str_repeat("-", 60) . "\n";

// Create sales prediction model
// Input: [day_of_week, is_holiday, weather_score, promotion_active]
// Output: [predicted_sales_category] (low, medium, high)
$salesModel = new Model();
$salesModel->addDense(new Dense(4, 6), 'relu');
$salesModel->addDense(new Dense(6, 3), 'softmax');

echo "Sales Prediction Model:\n";
echo $salesModel->summary();

$testCases = [
    'Friday, holiday, good weather, promotion' => [5, 1, 0.9, 1],
    'Monday, regular day, bad weather, no promotion' => [1, 0, 0.2, 0],
    'Saturday, regular day, good weather, promotion' => [6, 0, 0.8, 1],
];

echo "\nSales predictions (probabilities for low/medium/high):\n";
foreach ($testCases as $scenario => $features) {
    $input = Tensor::create($features);
    $output = $salesModel->predict($input);
    $probs = $output->toArray();
    
    echo "  $scenario\n";
    echo "    Low: " . number_format($probs[0] * 100, 1) . "% | ";
    echo "Medium: " . number_format($probs[1] * 100, 1) . "% | ";
    echo "High: " . number_format($probs[2] * 100, 1) . "%\n";
}

echo "\n";

echo "=== Examples completed successfully! ===\n";
