package neuralNetworkScratch;

import java.util.function.Function;

public class Layer {

    public enum ActivationFunctionType {
        SIGMOID,
        TANH
    }

    public enum InitialWeightType {
        RANDOM // only support random for the moment
    }

    double[][] weights;

    public final Function<Double, Double> activationFunction, activationFunctionDerivative;

    public Layer(int numberOfNeurons, int numberOfInputsPerNeuron) {
        this(ActivationFunctionType.SIGMOID, InitialWeightType.RANDOM, numberOfNeurons, numberOfInputsPerNeuron);
    }

    public Layer(ActivationFunctionType activationFunctionType, int numberOfNeurons, int numberOfInputsPerNeuron) {
        this(activationFunctionType, InitialWeightType.RANDOM, numberOfNeurons, numberOfInputsPerNeuron);
    }

    public Layer(ActivationFunctionType activationFunctionType, InitialWeightType initialWeightType, int numberOfNeurons, int numberOfInputsPerNeuron) {
        weights = new double[numberOfInputsPerNeuron][numberOfNeurons];

        for (int i = 0; i < numberOfInputsPerNeuron; ++i) {
            for (int j = 0; j < numberOfNeurons; ++j) {
                if (InitialWeightType.RANDOM == initialWeightType) {
                    weights[i][j] = (2 * Math.random()) - 1; // shift the range from 0-1 to -1 to 1
                }
            }
        }

        if (ActivationFunctionType.TANH == activationFunctionType) {
            activationFunction = NetMath::tanh;
            activationFunctionDerivative = NetMath::tanhDerivative;
        } else {
            activationFunction = NetMath::sigmoid;
            activationFunctionDerivative = NetMath::sigmoidDerivative;
        }
    }

    public void adjustWeights(double[][] adjustment) {
        this.weights = NetMath.matrixAdd(weights, adjustment);
    }

    @Override
    public String toString() {
        return MatrixUtility.matrixToString(weights);
    }
}
