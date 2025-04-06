package com.kaimueller_code.gnn_onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.Map;
import java.util.Random;

@SpringBootTest
class GnnOnnxApplicationTests {

	@Test
	void contextLoads() {
	}

	@Test
	void testModel() throws IOException, OrtException {

			String modelFilePath = new ClassPathResource("gnn_model.onnx").getFile().getPath();
		OrtEnvironment env = OrtEnvironment.getEnvironment();
		OrtSession session = env.createSession(modelFilePath);


		// Create random float array [2708, 1433]
		int[] floatShape = {2708, 1433};
		float[][] floatData = new float[2708][1433];
		Random rand = new Random();

		for (int i = 0; i < 2708; i++) {
			for (int j = 0; j < 1433; j++) {
				floatData[i][j] = rand.nextFloat();
			}
		}

		// Create random int array [2, 10556]
		int[] intShape = {2, 10556};
		long[][] intData = new long[2][10556];  // ONNX int tensor must be long[]

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 10556; j++) {
				intData[i][j] = rand.nextInt(2708);  // assume node indices range
			}
		}

		// Convert floatData to ONNX Tensor
		OnnxTensor floatTensor = OnnxTensor.createTensor(env, floatData);

		// Convert intData to ONNX Tensor
		OnnxTensor intTensor = OnnxTensor.createTensor(env, intData);

		// Just print shapes to confirm
		System.out.println("Float tensor shape: " + java.util.Arrays.toString(floatTensor.getInfo().getShape()));
		System.out.println("Int tensor shape: " + java.util.Arrays.toString(intTensor.getInfo().getShape()));

		OrtSession.Result res = session.run(Map.of("nodes", floatTensor, "edges", intTensor));
		System.out.println(res.get(0));

		floatTensor.close();
		intTensor.close();
		res.close();
	}

}
