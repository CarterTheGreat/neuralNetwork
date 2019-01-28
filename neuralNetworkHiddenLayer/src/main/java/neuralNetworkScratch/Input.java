package neuralNetworkScratch;

import neuralNetworkScratch.Network;

public class Input {

	public static void main(String[] args) {
		
		//layer with 1 neuron and 4 connections per neuron

		Layer hidden = new Layer(4, 3);
		Layer out = new Layer(1, 4);
		
		//Network with 1 layer
		Network network = new Network(hidden, out);
		
		double[][] inputs = new double[][]{
            {0, 0, 1},
            {1, 1, 1},
            {1, 0, 1},
            {0, 1, 1},
            {0, 0, 0}
    };

    double[][] outputs = new double[][]{
            {0},
            {1},
            {1},
            {0},
            {0}
    };
	/*	
		double[][] input = new double[][] {
			{1,0,1,0},
		};
		double[][] output = new double[][] {
			{1},
		};
		
		double[][] input1 = new double[][] {
			{1,1,1,1}
		};
		double[][] output1 = new double[][] {
			{1}
		};
		
		double[][] input2 = new double[][] {
			{0,0,1,0}
		};
		double[][] output2 = new double[][] {
			{0}
		};
		
		double[][] input3 = new double[][] {
			{0,0,0,0}
		};
		double[][] output3 = new double[][] {
			{0}
		};
		
		double[][] input4 = new double[][] {
			{0,1,1,1}
		};
		double[][] output4 = new double[][] {
			{0}
		};
	*/	
		System.out.println("Training");
		//Train network using above data sets 100000 times
		for(int i = 0; i<10000;i++) {
			network.train(inputs, outputs);
			//network.train(input1, output1);
			//network.train(input2, output2);
			//network.train(input3, output3);
			//network.train(input4, output4);
		}
		
		System.out.println("Hidden weights: \n"+MatrixUtility.matrixToString(hidden.weights));
		System.out.println("Out weights: \n"+ MatrixUtility.matrixToString(out.weights));
		
		System.out.println("Predictions");		
		//Get Answers 
		predict(new double[][]{{1,0,1}}, network);
		predict(new double[][]{{0,1,1}}, network);
		predict(new double[][]{{1,1,1}}, network);
		
		
	}
	
	public static void predict(double[][] testInput, Network network) {
		network.process(testInput);
		
		System.out.println("Prediction on data "
				+ testInput[0][0] + " "
				+ testInput[0][1] + " "
				+ testInput[0][2] + " "
				//+ testInput[0][3] + " -> "
				+ network.getOutput()[0][0] + " ("+Math.round(network.getOutput()[0][0])+") "+", Expected: " + testInput[0][0]);
	}
	
	
}
