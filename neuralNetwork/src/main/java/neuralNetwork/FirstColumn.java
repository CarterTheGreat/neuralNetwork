package neuralNetwork;

public class FirstColumn {

	
	
	public static void main(String[] args) {
		
		//Create hidden layer - Layer object
		Layer layer1 = new Layer(1,3);
		
		//Create output layer
		Network network = new Network(layer1);
		
		//Define inputs 
		double[][] inputs = new double[][]{
            {0, 0, 1},
            {1, 1, 1},
            {1, 0, 1},
            {0, 1, 1}
		};

		double[][] outputs = new double[][]{
            {0},
            {1},
            {1},
            {0}
    	};
		
		System.out.println("Training the neural net...");
        network.train(inputs, outputs, 10000);
        System.out.println("Finished training");

        System.out.println("Layer 1 weights");
        System.out.println(layer1);
		
		
        //Calculate predictions for unknown data
        
        //1,0,0
        predict(new double[][]{{1, 0, 0}}, network);

        // 0, 1, 0
        predict(new double[][]{{0, 1, 0}}, network);

        // 1, 1, 0
        predict(new double[][]{{1, 1, 0}}, network);
		
	}
	
	public static void predict(double[][] testInput, Network network) {
		network.process(testInput);
		
		System.out.println("Prediction on data "
				+ testInput[0][0] + " "
				+ testInput[0][1] + " "
				+ testInput[0][2] + " -> "
				+ network.getOutput()[0][0] + ", Expected: " + testInput[0][0]);
	}
	
}
