<?php
namespace Opencart\System\Library\Nn;

use Opencart\System\Library\Tensor\Tensor;

/**
 * Dense (Fully Connected) Layer
 */
class Dense {
    private Tensor $weights;
    private Tensor $bias;
    private int $inputSize;
    private int $outputSize;
    
    /**
     * Constructor
     * 
     * @param int $inputSize Number of input features
     * @param int $outputSize Number of output features
     * @param bool $useBias Whether to use bias
     */
    public function __construct(int $inputSize, int $outputSize, bool $useBias = true) {
        $this->inputSize = $inputSize;
        $this->outputSize = $outputSize;
        
        // Initialize weights with Xavier/Glorot uniform initialization
        $limit = sqrt(6.0 / ($inputSize + $outputSize));
        $weightsData = [];
        for ($i = 0; $i < $inputSize * $outputSize; $i++) {
            $weightsData[] = (mt_rand() / mt_getrandmax()) * 2 * $limit - $limit;
        }
        $this->weights = new Tensor($weightsData, [$inputSize, $outputSize]);
        
        // Initialize bias to zeros
        if ($useBias) {
            $this->bias = Tensor::zeros([$outputSize]);
        } else {
            $this->bias = Tensor::zeros([$outputSize]);
        }
    }
    
    /**
     * Forward pass
     * 
     * @param Tensor $input Input tensor (batch_size x input_size or input_size)
     * @return Tensor Output tensor
     */
    public function forward(Tensor $input): Tensor {
        $inputShape = $input->getShape();
        
        // Handle both batched and single input
        if (count($inputShape) === 1) {
            // Single input vector
            if ($inputShape[0] !== $this->inputSize) {
                throw new \InvalidArgumentException("Input size mismatch");
            }
            
            // Reshape input to column vector for matmul
            $inputReshaped = new Tensor($input->getData(), [1, $this->inputSize]);
            $output = $inputReshaped->matmul($this->weights);
            
            // Add bias
            $outputData = $output->getData();
            $biasData = $this->bias->getData();
            for ($i = 0; $i < $this->outputSize; $i++) {
                $outputData[$i] += $biasData[$i];
            }
            
            return new Tensor($outputData, [$this->outputSize]);
        } else if (count($inputShape) === 2) {
            // Batched input
            $batchSize = $inputShape[0];
            if ($inputShape[1] !== $this->inputSize) {
                throw new \InvalidArgumentException("Input size mismatch");
            }
            
            $output = $input->matmul($this->weights);
            
            // Add bias to each sample in batch
            $outputData = $output->getData();
            $biasData = $this->bias->getData();
            for ($i = 0; $i < $batchSize; $i++) {
                for ($j = 0; $j < $this->outputSize; $j++) {
                    $outputData[$i * $this->outputSize + $j] += $biasData[$j];
                }
            }
            
            return new Tensor($outputData, [$batchSize, $this->outputSize]);
        } else {
            throw new \InvalidArgumentException("Input must be 1D or 2D tensor");
        }
    }
    
    /**
     * Get weights
     * 
     * @return Tensor
     */
    public function getWeights(): Tensor {
        return $this->weights;
    }
    
    /**
     * Get bias
     * 
     * @return Tensor
     */
    public function getBias(): Tensor {
        return $this->bias;
    }
    
    /**
     * Set weights
     * 
     * @param Tensor $weights
     */
    public function setWeights(Tensor $weights): void {
        if ($weights->getShape() !== [$this->inputSize, $this->outputSize]) {
            throw new \InvalidArgumentException("Weights shape mismatch");
        }
        $this->weights = $weights;
    }
    
    /**
     * Set bias
     * 
     * @param Tensor $bias
     */
    public function setBias(Tensor $bias): void {
        if ($bias->getShape() !== [$this->outputSize]) {
            throw new \InvalidArgumentException("Bias shape mismatch");
        }
        $this->bias = $bias;
    }
    
    /**
     * Serialize layer to array
     * 
     * @return array<string, mixed>
     */
    public function toArray(): array {
        return [
            'type' => 'dense',
            'input_size' => $this->inputSize,
            'output_size' => $this->outputSize,
            'weights' => json_decode($this->weights->toJson(), true),
            'bias' => json_decode($this->bias->toJson(), true)
        ];
    }
    
    /**
     * Create layer from array
     * 
     * @param array<string, mixed> $data
     * @return self
     */
    public static function fromArray(array $data): self {
        $layer = new self($data['input_size'], $data['output_size']);
        $layer->setWeights(Tensor::fromJson(json_encode($data['weights'])));
        $layer->setBias(Tensor::fromJson(json_encode($data['bias'])));
        return $layer;
    }
}
