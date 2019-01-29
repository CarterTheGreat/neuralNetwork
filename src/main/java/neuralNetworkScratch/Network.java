package neuralNetworkScratch;

import java.util.function.Function;
import neuralNetworkScratch.NetMath.*;


/**
 * https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a#.9kcfharq6
 * http://stevenmiller888.github.io/mind-how-to-build-a-neural-network-part-2/
 */
public class Network {

    private final Layer layer1, layer2;
    private double[][] outputLayer1;
    private double[][] outputLayer2;
    private final double learningRate;

    public Network(Layer layer1, Layer layer2) {
        this(layer1, layer2, 0.1);
    }

    public Network(Layer layer1, Layer layer2, double learningRate) {
        this.layer1 = layer1;
        this.layer2 = layer2;
        this.learningRate = learningRate;
    }

    /**
     * Forward propagation
     * <p>
     * Output of neuron = 1 / (1 + e^(-(sum(weight, input)))
     *
     * @param inputs
     */
    public void process(double[][] inputs) {
        outputLayer1 = MatrixUtility.applyToFunction(NetMath.matrixMultiply(inputs, layer1.weights), layer1.activationFunction); // 4x4
        outputLayer2 = MatrixUtility.applyToFunction(NetMath.matrixMultiply(outputLayer1, layer2.weights), layer2.activationFunction); // 4x1
    }

    public void train(double[][] inputs, double[][] outputs) {
            // pass the training set through the network
            process(inputs); // 4x3

            // adjust weights by error * input * output * (1 - output)

            // calculate the error for layer 2
            // (the difference between the desired output and predicted output for each of the training inputs)
            double[][] errorLayer2 = NetMath.matrixSubtract(outputs, outputLayer2); // 4x1
            double[][] deltaLayer2 = NetMath.scalarMultiply(errorLayer2, MatrixUtility.applyToFunction(outputLayer2, layer2.activationFunctionDerivative)); // 4x1

            // calculate the error for layer 1
            // (by looking at the weights in layer 1, we can determine by how much layer 1 contributed to the error in layer 2)

            double[][] errorLayer1 = NetMath.matrixMultiply(deltaLayer2, NetMath.matrixTranspose(layer2.weights)); // 4x4
            double[][] deltaLayer1 = NetMath.scalarMultiply(errorLayer1, MatrixUtility.applyToFunction(outputLayer1, layer1.activationFunctionDerivative)); // 4x4

            // Calculate how much to adjust the weights by
            // Since we're dealing with matrices, we handle the division by multiplying the delta output sum with the inputs' transpose!

            double[][] adjustmentLayer1 = NetMath.matrixMultiply(NetMath.matrixTranspose(inputs), deltaLayer1); // 4x4
            double[][] adjustmentLayer2 = NetMath.matrixMultiply(NetMath.matrixTranspose(outputLayer1), deltaLayer2); // 4x1

            adjustmentLayer1 = MatrixUtility.applyToFunction(adjustmentLayer1, (x) -> learningRate * x);
            adjustmentLayer2 = MatrixUtility.applyToFunction(adjustmentLayer2, (x) -> learningRate * x);

            // adjust the weights
            this.layer1.adjustWeights(adjustmentLayer1);
            this.layer2.adjustWeights(adjustmentLayer2);

            System.out.println("Hidden adjust: \n"+MatrixUtility.matrixToString(adjustmentLayer1));
    		System.out.println("Out adjust: \n"+MatrixUtility.matrixToString(adjustmentLayer2));
    		
            
            // if you only had one layer
            // synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
            // double[][] errorLayer1 = NNMath.NetMath.matrixSubtract(outputs, outputLayer1);
            // double[][] deltaLayer1 = NNMath.NetMath.matrixMultiply(errorLayer1, MatrixUtility.applyToFunction(outputLayer1, NNMath::sigmoidDerivative));
            // double[][] adjustmentLayer1 = NNMath.NetMath.matrixMultiply(NNMath.NetMath.matrixTranspose(inputs), deltaLayer1);

           
    }

    public double[][] getOutput() {
        return outputLayer2;
    }

    @Override
    public String toString() {
        String result = "Layer 1\n";
        result += layer1.toString();
        result += "Layer 2\n";
        result += layer2.toString();

        if (outputLayer1 != null) {
            result += "Layer 1 output\n";
            result += MatrixUtility.matrixToString(outputLayer1);
        }

        if (outputLayer2 != null) {
            result += "Layer 2 output\n";
            result += MatrixUtility.matrixToString(outputLayer2);
        }

        return result;
    }
}
