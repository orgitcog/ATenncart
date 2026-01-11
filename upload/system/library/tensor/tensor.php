<?php
namespace Opencart\System\Library\Tensor;

/**
 * Tensor class - ATen-inspired tensor operations for PHP
 * 
 * Provides basic tensor operations similar to PyTorch's ATen library
 */
class Tensor {
    /**
     * @var array<int, int> Shape of the tensor
     */
    private array $shape;
    
    /**
     * @var array<mixed> Data storage (flat array)
     */
    private array $data;
    
    /**
     * @var int Total number of elements
     */
    private int $size;
    
    /**
     * Constructor
     * 
     * @param array<mixed> $data Input data (nested arrays)
     * @param array<int, int>|null $shape Optional shape specification
     */
    public function __construct(array $data, ?array $shape = null) {
        if ($shape === null) {
            $this->shape = $this->inferShape($data);
        } else {
            $this->shape = $shape;
        }
        
        $this->data = $this->flatten($data);
        $this->size = count($this->data);
        
        // Validate shape matches data size
        $expectedSize = array_product($this->shape);
        if ($expectedSize !== $this->size) {
            throw new \InvalidArgumentException("Shape {$expectedSize} does not match data size {$this->size}");
        }
    }
    
    /**
     * Create a tensor from data
     * 
     * @param array<mixed> $data
     * @return self
     */
    public static function create(array $data): self {
        return new self($data);
    }
    
    /**
     * Create a tensor filled with zeros
     * 
     * @param array<int, int> $shape
     * @return self
     */
    public static function zeros(array $shape): self {
        $size = array_product($shape);
        $data = array_fill(0, $size, 0.0);
        return new self($data, $shape);
    }
    
    /**
     * Create a tensor filled with ones
     * 
     * @param array<int, int> $shape
     * @return self
     */
    public static function ones(array $shape): self {
        $size = array_product($shape);
        $data = array_fill(0, $size, 1.0);
        return new self($data, $shape);
    }
    
    /**
     * Create a tensor with random values between 0 and 1
     * 
     * @param array<int, int> $shape
     * @return self
     */
    public static function rand(array $shape): self {
        $size = array_product($shape);
        $data = [];
        for ($i = 0; $i < $size; $i++) {
            $data[] = mt_rand() / mt_getrandmax();
        }
        return new self($data, $shape);
    }
    
    /**
     * Get tensor shape
     * 
     * @return array<int, int>
     */
    public function getShape(): array {
        return $this->shape;
    }
    
    /**
     * Get tensor data as flat array
     * 
     * @return array<mixed>
     */
    public function getData(): array {
        return $this->data;
    }
    
    /**
     * Get tensor data as nested array matching shape
     * 
     * @return array<mixed>
     */
    public function toArray(): array {
        return $this->reshape($this->data, $this->shape);
    }
    
    /**
     * Get tensor size (total elements)
     * 
     * @return int
     */
    public function getSize(): int {
        return $this->size;
    }
    
    /**
     * Element-wise addition
     * 
     * @param self|float $other
     * @return self
     */
    public function add($other): self {
        if ($other instanceof self) {
            if ($this->shape !== $other->shape) {
                throw new \InvalidArgumentException("Shape mismatch for addition");
            }
            $result = [];
            for ($i = 0; $i < $this->size; $i++) {
                $result[] = $this->data[$i] + $other->data[$i];
            }
            return new self($result, $this->shape);
        } else {
            // Scalar addition
            $result = array_map(fn($x) => $x + $other, $this->data);
            return new self($result, $this->shape);
        }
    }
    
    /**
     * Element-wise subtraction
     * 
     * @param self|float $other
     * @return self
     */
    public function sub($other): self {
        if ($other instanceof self) {
            if ($this->shape !== $other->shape) {
                throw new \InvalidArgumentException("Shape mismatch for subtraction");
            }
            $result = [];
            for ($i = 0; $i < $this->size; $i++) {
                $result[] = $this->data[$i] - $other->data[$i];
            }
            return new self($result, $this->shape);
        } else {
            // Scalar subtraction
            $result = array_map(fn($x) => $x - $other, $this->data);
            return new self($result, $this->shape);
        }
    }
    
    /**
     * Element-wise multiplication
     * 
     * @param self|float $other
     * @return self
     */
    public function mul($other): self {
        if ($other instanceof self) {
            if ($this->shape !== $other->shape) {
                throw new \InvalidArgumentException("Shape mismatch for multiplication");
            }
            $result = [];
            for ($i = 0; $i < $this->size; $i++) {
                $result[] = $this->data[$i] * $other->data[$i];
            }
            return new self($result, $this->shape);
        } else {
            // Scalar multiplication
            $result = array_map(fn($x) => $x * $other, $this->data);
            return new self($result, $this->shape);
        }
    }
    
    /**
     * Element-wise division
     * 
     * @param self|float $other
     * @return self
     */
    public function div($other): self {
        if ($other instanceof self) {
            if ($this->shape !== $other->shape) {
                throw new \InvalidArgumentException("Shape mismatch for division");
            }
            $result = [];
            for ($i = 0; $i < $this->size; $i++) {
                if ($other->data[$i] == 0) {
                    throw new \DivisionByZeroError("Division by zero");
                }
                $result[] = $this->data[$i] / $other->data[$i];
            }
            return new self($result, $this->shape);
        } else {
            if ($other == 0) {
                throw new \DivisionByZeroError("Division by zero");
            }
            // Scalar division
            $result = array_map(fn($x) => $x / $other, $this->data);
            return new self($result, $this->shape);
        }
    }
    
    /**
     * Matrix multiplication (dot product)
     * 
     * @param self $other
     * @return self
     */
    public function matmul(self $other): self {
        // Support 2D matrix multiplication
        if (count($this->shape) !== 2 || count($other->shape) !== 2) {
            throw new \InvalidArgumentException("matmul requires 2D tensors");
        }
        
        $m = $this->shape[0];
        $n = $this->shape[1];
        $p = $other->shape[1];
        
        if ($n !== $other->shape[0]) {
            throw new \InvalidArgumentException("Inner dimensions must match for matmul");
        }
        
        $result = [];
        for ($i = 0; $i < $m; $i++) {
            for ($j = 0; $j < $p; $j++) {
                $sum = 0.0;
                for ($k = 0; $k < $n; $k++) {
                    $sum += $this->data[$i * $n + $k] * $other->data[$k * $p + $j];
                }
                $result[] = $sum;
            }
        }
        
        return new self($result, [$m, $p]);
    }
    
    /**
     * Transpose (2D only)
     * 
     * @return self
     */
    public function transpose(): self {
        if (count($this->shape) !== 2) {
            throw new \InvalidArgumentException("transpose requires 2D tensor");
        }
        
        $rows = $this->shape[0];
        $cols = $this->shape[1];
        $result = [];
        
        for ($j = 0; $j < $cols; $j++) {
            for ($i = 0; $i < $rows; $i++) {
                $result[] = $this->data[$i * $cols + $j];
            }
        }
        
        return new self($result, [$cols, $rows]);
    }
    
    /**
     * Apply function element-wise
     * 
     * @param callable $func
     * @return self
     */
    public function apply(callable $func): self {
        $result = array_map($func, $this->data);
        return new self($result, $this->shape);
    }
    
    /**
     * Sum all elements
     * 
     * @return float
     */
    public function sum(): float {
        return array_sum($this->data);
    }
    
    /**
     * Mean of all elements
     * 
     * @return float
     */
    public function mean(): float {
        return $this->sum() / $this->size;
    }
    
    /**
     * Maximum value
     * 
     * @return float
     */
    public function max(): float {
        return max($this->data);
    }
    
    /**
     * Minimum value
     * 
     * @return float
     */
    public function min(): float {
        return min($this->data);
    }
    
    /**
     * Infer shape from nested array
     * 
     * @param array<mixed> $data
     * @return array<int, int>
     */
    private function inferShape(array $data): array {
        $shape = [];
        $current = $data;
        
        while (is_array($current)) {
            $shape[] = count($current);
            if (empty($current)) {
                break;
            }
            $current = reset($current);
        }
        
        return $shape;
    }
    
    /**
     * Flatten nested array
     * 
     * @param array<mixed> $data
     * @return array<mixed>
     */
    private function flatten(array $data): array {
        $result = [];
        array_walk_recursive($data, function($value) use (&$result) {
            $result[] = (float)$value;
        });
        return $result;
    }
    
    /**
     * Reshape flat array to nested structure
     * 
     * @param array<mixed> $data
     * @param array<int, int> $shape
     * @return array<mixed>
     */
    private function reshape(array $data, array $shape): array {
        if (count($shape) === 1) {
            return $data;
        }
        
        $result = [];
        $stride = array_product(array_slice($shape, 1));
        
        for ($i = 0; $i < $shape[0]; $i++) {
            $slice = array_slice($data, $i * $stride, $stride);
            $result[] = $this->reshape($slice, array_slice($shape, 1));
        }
        
        return $result;
    }
    
    /**
     * Serialize tensor to JSON
     * 
     * @return string
     */
    public function toJson(): string {
        return json_encode([
            'shape' => $this->shape,
            'data' => $this->data
        ]);
    }
    
    /**
     * Create tensor from JSON
     * 
     * @param string $json
     * @return self
     */
    public static function fromJson(string $json): self {
        $obj = json_decode($json, true);
        if (!isset($obj['shape']) || !isset($obj['data'])) {
            throw new \InvalidArgumentException("Invalid JSON format");
        }
        return new self($obj['data'], $obj['shape']);
    }
}
