package neuralNetwork;


import static neuralNetwork.MatrixUtil.apply;
import static neuralNetwork.NetworkMath.*;

public class Network {

	private final Layer layer1;
	private double[][] outputLayer1;
	
	public Network(Layer layer1) {
		this.layer1 = layer1;
	}
	
	
	/*
	 * Forward propagation
	 * 
	 * Output of neuron = 1 / (1 + e^(-(sum(weight, input)))
	 */
	
	public void process(double[][] inputs) {
		outputLayer1 = apply(matrixMultiply(inputs, layer1.weights), layer1.activationFunction);
	}
	
	public void train(double[][] inputs, double[][] outputs, int numberOfTrainingIterations) {
		
		for(int i = 0; i < numberOfTrainingIterations; i++) {
			//Run training sets through network
			process(inputs);
			
			//Adjust weight by error * input * output  * (1 - output)
			double[][] errorLayer1 = matrixSubtract(outputs, outputLayer1);
			double[][] deltaLayer1 =  scalarMultiply(errorLayer1, apply(outputLayer1, layer1.activationFunctionDerivative));
			
			// Calculate how much to adjust the weights by
            // Since were dealing with matrices, we handle the division by multiplying the delta output sum with the inputs' transpose!

			double[][] adjustmentLayer1 = matrixMultiply(matrixTranspose(inputs),deltaLayer1);
			
			//Adjust weights
			
			this.layer1.adjustWeights(adjustmentLayer1);
			
			 if (i % 1000 == 0) {
	                System.out.println(" Training iteration " + i + " of " + numberOfTrainingIterations);
	            }
			
		}
		
	}
	
	public double[][] getOutput(){
		return outputLayer1;
	}
	
	public String toString() {
		String result = "Layer 1\n";
		result += layer1.toString();

		if (outputLayer1 != null) {
            result += "Layer 1 output\n";
            result += MatrixUtil.matrixToString(outputLayer1);
        }
		
		return result;
		
	}

}
