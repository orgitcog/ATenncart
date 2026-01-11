<?php
namespace Opencart\Catalog\Controller\Extension\TensorLogic\Api;

use Opencart\System\Library\Nn\Dense;
use Opencart\System\Library\Nn\Model;
use Opencart\System\Library\Tensor\Tensor;

/**
 * Tensor Logic API Controller
 *
 * Provides REST API endpoints for tensor operations and neural network inference
 */
class TensorLogic extends \Opencart\System\Engine\Controller {
	/**
	 * API endpoint: Create a tensor
	 *
	 * POST /index.php?route=extension/tensor_logic/api/tensor_logic.create
	 * Body: {"data": [[1, 2], [3, 4]]}
	 */
	public function create(): void {
		$this->response->addHeader('Content-Type: application/json');

		try {
			$json = file_get_contents('php://input');
			$input = json_decode($json, true);

			if (!isset($input['data'])) {
				throw new \InvalidArgumentException("Missing 'data' field");
			}

			$tensor = Tensor::create($input['data']);

			$this->response->setOutput(json_encode([
				'success' => true,
				'tensor'  => [
					'shape' => $tensor->getShape(),
					'size'  => $tensor->getSize(),
					'data'  => $tensor->toArray()
				]
			]));
		} catch (\Exception $e) {
			$this->response->setOutput(json_encode([
				'success' => false,
				'error'   => $e->getMessage()
			]));
		}
	}

	/**
	 * API endpoint: Add tensors
	 *
	 * POST /index.php?route=extension/tensor_logic/api/tensor_logic.add
	 * Body: {"a": [[1, 2]], "b": [[3, 4]]}
	 */
	public function add(): void {
		$this->response->addHeader('Content-Type: application/json');

		try {
			$json = file_get_contents('php://input');
			$input = json_decode($json, true);

			if (!isset($input['a']) || !isset($input['b'])) {
				throw new \InvalidArgumentException("Missing 'a' or 'b' field");
			}

			$tensorA = Tensor::create($input['a']);
			$tensorB = Tensor::create($input['b']);
			$result = $tensorA->add($tensorB);

			$this->response->setOutput(json_encode([
				'success' => true,
				'result'  => [
					'shape' => $result->getShape(),
					'data'  => $result->toArray()
				]
			]));
		} catch (\Exception $e) {
			$this->response->setOutput(json_encode([
				'success' => false,
				'error'   => $e->getMessage()
			]));
		}
	}

	/**
	 * API endpoint: Multiply tensors
	 *
	 * POST /index.php?route=extension/tensor_logic/api/tensor_logic.multiply
	 * Body: {"a": [[1, 2]], "b": [[3, 4]]}
	 */
	public function multiply(): void {
		$this->response->addHeader('Content-Type: application/json');

		try {
			$json = file_get_contents('php://input');
			$input = json_decode($json, true);

			if (!isset($input['a']) || !isset($input['b'])) {
				throw new \InvalidArgumentException("Missing 'a' or 'b' field");
			}

			$tensorA = Tensor::create($input['a']);
			$tensorB = Tensor::create($input['b']);
			$result = $tensorA->mul($tensorB);

			$this->response->setOutput(json_encode([
				'success' => true,
				'result'  => [
					'shape' => $result->getShape(),
					'data'  => $result->toArray()
				]
			]));
		} catch (\Exception $e) {
			$this->response->setOutput(json_encode([
				'success' => false,
				'error'   => $e->getMessage()
			]));
		}
	}

	/**
	 * API endpoint: Matrix multiplication
	 *
	 * POST /index.php?route=extension/tensor_logic/api/tensor_logic.matmul
	 * Body: {"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]}
	 */
	public function matmul(): void {
		$this->response->addHeader('Content-Type: application/json');

		try {
			$json = file_get_contents('php://input');
			$input = json_decode($json, true);

			if (!isset($input['a']) || !isset($input['b'])) {
				throw new \InvalidArgumentException("Missing 'a' or 'b' field");
			}

			$tensorA = Tensor::create($input['a']);
			$tensorB = Tensor::create($input['b']);
			$result = $tensorA->matmul($tensorB);

			$this->response->setOutput(json_encode([
				'success' => true,
				'result'  => [
					'shape' => $result->getShape(),
					'data'  => $result->toArray()
				]
			]));
		} catch (\Exception $e) {
			$this->response->setOutput(json_encode([
				'success' => false,
				'error'   => $e->getMessage()
			]));
		}
	}

	/**
	 * API endpoint: Neural network inference
	 *
	 * POST /index.php?route=extension/tensor_logic/api/tensor_logic.predict
	 * Body: {"model": {...}, "input": [1, 2, 3]}
	 */
	public function predict(): void {
		$this->response->addHeader('Content-Type: application/json');

		try {
			$json = file_get_contents('php://input');
			$input = json_decode($json, true);

			if (!isset($input['model']) || !isset($input['input'])) {
				throw new \InvalidArgumentException("Missing 'model' or 'input' field");
			}

			// Load model from JSON
			$model = Model::load(json_encode($input['model']));

			// Create input tensor
			$inputTensor = Tensor::create($input['input']);

			// Run inference
			$output = $model->predict($inputTensor);

			$this->response->setOutput(json_encode([
				'success' => true,
				'output'  => [
					'shape' => $output->getShape(),
					'data'  => $output->toArray()
				]
			]));
		} catch (\Exception $e) {
			$this->response->setOutput(json_encode([
				'success' => false,
				'error'   => $e->getMessage()
			]));
		}
	}

	/**
	 * API endpoint: Create example neural network model
	 *
	 * GET /index.php?route=extension/tensor_logic/api/tensor_logic.createModel
	 * Query params: ?input_size=3&hidden_size=5&output_size=2
	 */
	public function createModel(): void {
		$this->response->addHeader('Content-Type: application/json');

		try {
			$inputSize = isset($this->request->get['input_size']) ? (int)$this->request->get['input_size'] : 3;
			$hiddenSize = isset($this->request->get['hidden_size']) ? (int)$this->request->get['hidden_size'] : 5;
			$outputSize = isset($this->request->get['output_size']) ? (int)$this->request->get['output_size'] : 2;

			// Create a simple model
			$model = new Model();
			$model->addDense(new Dense($inputSize, $hiddenSize), 'relu');
			$model->addDense(new Dense($hiddenSize, $outputSize), 'softmax');

			$modelJson = $model->save();

			$this->response->setOutput(json_encode([
				'success' => true,
				'model'   => json_decode($modelJson, true),
				'summary' => $model->summary()
			]));
		} catch (\Exception $e) {
			$this->response->setOutput(json_encode([
				'success' => false,
				'error'   => $e->getMessage()
			]));
		}
	}

	/**
	 * API endpoint: Get API info
	 *
	 * GET /index.php?route=extension/tensor_logic/api/tensor_logic
	 */
	public function index(): void {
		$this->response->addHeader('Content-Type: application/json');

		$this->response->setOutput(json_encode([
			'name'        => 'Tensor Logic API',
			'version'     => '1.0.0',
			'description' => 'ATen & nn inspired tensor operations and neural networks for OpenCart',
			'endpoints'   => [
				'create'      => 'POST /index.php?route=extension/tensor_logic/api/tensor_logic.create',
				'add'         => 'POST /index.php?route=extension/tensor_logic/api/tensor_logic.add',
				'multiply'    => 'POST /index.php?route=extension/tensor_logic/api/tensor_logic.multiply',
				'matmul'      => 'POST /index.php?route=extension/tensor_logic/api/tensor_logic.matmul',
				'predict'     => 'POST /index.php?route=extension/tensor_logic/api/tensor_logic.predict',
				'createModel' => 'GET /index.php?route=extension/tensor_logic/api/tensor_logic.createModel'
			],
			'documentation' => 'https://tensor-logic.org'
		]));
	}
}
