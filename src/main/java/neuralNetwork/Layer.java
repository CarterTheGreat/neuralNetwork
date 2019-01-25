package neuralNetwork;


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
		 this(ActivationFunctionType.SIGMOID, InitialWeightType.RANDOM, numberOfNeurons, numberOfInputsPerNeuron);
	 }
	 
	 public Layer(ActivationFunctionType activationFunctionType, InitialWeightType initialWeightType, int numberOfNeurons, int numberOfInputsPerNeuron ) {


	}
	
}
