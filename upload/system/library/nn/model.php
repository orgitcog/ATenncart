<?php
namespace Opencart\System\Library\Nn;

use Opencart\System\Library\Tensor\Tensor;

/**
 * Sequential Neural Network Model
 */
class Model {
    /**
     * @var array<int, array{layer: Dense|string, activation: string|null}> Layers
     */
    private array $layers = [];
    
    /**
     * Add a dense layer
     * 
     * @param Dense $layer
     * @param string|null $activation Activation function name (relu, sigmoid, tanh, softmax, leaky_relu)
     * @return self
     */
    public function addDense(Dense $layer, ?string $activation = null): self {
        $this->layers[] = [
            'layer' => $layer,
            'activation' => $activation
        ];
        return $this;
    }
    
    /**
     * Forward pass through the network
     * 
     * @param Tensor $input
     * @return Tensor
     */
    public function forward(Tensor $input): Tensor {
        $output = $input;
        
        foreach ($this->layers as $layerConfig) {
            $layer = $layerConfig['layer'];
            $activation = $layerConfig['activation'];
            
            // Apply layer
            if ($layer instanceof Dense) {
                $output = $layer->forward($output);
            }
            
            // Apply activation if specified
            if ($activation !== null) {
                $output = $this->applyActivation($output, $activation);
            }
        }
        
        return $output;
    }
    
    /**
     * Predict (alias for forward)
     * 
     * @param Tensor $input
     * @return Tensor
     */
    public function predict(Tensor $input): Tensor {
        return $this->forward($input);
    }
    
    /**
     * Apply activation function
     * 
     * @param Tensor $input
     * @param string $activation
     * @return Tensor
     */
    private function applyActivation(Tensor $input, string $activation): Tensor {
        switch (strtolower($activation)) {
            case 'relu':
                return Activation::relu($input);
            case 'sigmoid':
                return Activation::sigmoid($input);
            case 'tanh':
                return Activation::tanh($input);
            case 'softmax':
                return Activation::softmax($input);
            case 'leaky_relu':
                return Activation::leakyRelu($input);
            default:
                throw new \InvalidArgumentException("Unknown activation: {$activation}");
        }
    }
    
    /**
     * Get number of layers
     * 
     * @return int
     */
    public function getLayerCount(): int {
        return count($this->layers);
    }
    
    /**
     * Save model to JSON string
     * 
     * @return string
     */
    public function save(): string {
        $data = [
            'version' => '1.0',
            'layers' => []
        ];
        
        foreach ($this->layers as $layerConfig) {
            $layer = $layerConfig['layer'];
            $activation = $layerConfig['activation'];
            
            if ($layer instanceof Dense) {
                $layerData = $layer->toArray();
                $layerData['activation'] = $activation;
                $data['layers'][] = $layerData;
            }
        }
        
        return json_encode($data, JSON_PRETTY_PRINT);
    }
    
    /**
     * Load model from JSON string
     * 
     * @param string $json
     * @return self
     */
    public static function load(string $json): self {
        $data = json_decode($json, true);
        
        if (!isset($data['layers'])) {
            throw new \InvalidArgumentException("Invalid model format");
        }
        
        $model = new self();
        
        foreach ($data['layers'] as $layerData) {
            if ($layerData['type'] === 'dense') {
                $layer = Dense::fromArray($layerData);
                $activation = $layerData['activation'] ?? null;
                $model->addDense($layer, $activation);
            }
        }
        
        return $model;
    }
    
    /**
     * Get model summary as string
     * 
     * @return string
     */
    public function summary(): string {
        $summary = "Model Summary:\n";
        $summary .= str_repeat("=", 60) . "\n";
        $summary .= sprintf("%-20s %-20s %-15s\n", "Layer", "Output Shape", "Activation");
        $summary .= str_repeat("-", 60) . "\n";
        
        $layerNum = 1;
        foreach ($this->layers as $layerConfig) {
            $layer = $layerConfig['layer'];
            $activation = $layerConfig['activation'] ?? 'none';
            
            if ($layer instanceof Dense) {
                $outputShape = "[" . $layer->getBias()->getShape()[0] . "]";
                $summary .= sprintf("%-20s %-20s %-15s\n", 
                    "Dense_{$layerNum}", 
                    $outputShape, 
                    $activation
                );
            }
            $layerNum++;
        }
        
        $summary .= str_repeat("=", 60) . "\n";
        return $summary;
    }
}
