
#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "isolation_forest_model.h"

const char* testNames[] = {
  "normal",
  "imbalance",
  "loose_mount",
  "blocked_airflow",
  "near_threshold"
};

const double expectedScores[] = {
   0.079781389062,
  -0.264125659260,
  -0.050946495109,
  -0.120439482034,
   0.031594415840
};


const float testFeatures[5][9] = {
  {
    9.20246601f, 9.41544914f, 0.0946171135f,
    -0.542649686f, 1.02314413f,
    1.75183923e-05f, 3.06829061e-05f,
    0.000114947921f, 5.20299727e-06f
  },
  {
    9.22028255f, 10.2017479f, 0.476679176f,
    -1.06278789f, 1.10644627f,
    2.3672108e-05f, 0.00842424296f,
    0.0002154182f, 1.93471096e-05f
  },
  {
    9.20095539f, 9.43881798f, 0.0984279513f,
    -0.319408685f, 1.02585196f,
    1.36622339e-05f, 3.55874326e-05f,
    0.000102246035f, 1.50278029e-05f
  },
  {
    9.20150375f, 9.5297699f, 0.105654269f,
    -0.122803062f, 1.03567529f,
    2.14325828e-05f, 4.51037995e-05f,
    0.000124596321f, 1.34864222e-05f
  },
  {
    9.20707703f, 9.45608997f, 0.0953597873f,
    -0.593912542f, 1.02704573f,
    1.4329913e-05f, 2.63326874e-05f,
    0.000117720119f, 6.06476169e-06f
  }
};

// Expected isolation path length for n samples
double averagePathLength(int n) {
  if (n <= 1) return 0.0;
  if (n == 2) return 1.0;

  const double eulerGamma = 0.5772156649015329;

  return 2.0 * (log((double)n - 1.0) + eulerGamma)
       - 2.0 * (n - 1.0) / n;
}

// Calculate the decision_function score
double isolationForestScore(const float features[9]) {
  double totalPathLength = 0.0;

  for (int t = 0; t < IF_TREE_COUNT; t++) {
    uint16_t treeOffset =
      pgm_read_word(&IF_TREE_OFFSETS[t]);

    int nodeIndex = 0;
    int depth = 0;

    while (true) {
      IFNode node;

      memcpy_P(
        &node,
        &IF_NODES[treeOffset + nodeIndex],
        sizeof(IFNode)
      );

      if (node.left == -1) {
        totalPathLength +=
          depth + averagePathLength(node.n_samples);
        break;
      }

      if ((double)features[node.feature] <= node.threshold) {
        nodeIndex = node.left;
      } else {
        nodeIndex = node.right;
      }

      depth++;
    }
  }

  double denominator =
    IF_TREE_COUNT * averagePathLength(IF_MAX_SAMPLES);

  double rawScore =
    -pow(2.0, -totalPathLength / denominator);

  // Match scikit-learn decision_function
  return rawScore - IF_OFFSET;
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println();
  Serial.println("ESP32 Isolation Forest Verification");
  Serial.println("---------------------------------");

  Serial.printf("Trees: %d\n", IF_TREE_COUNT);
  Serial.printf("Features: %d\n", IF_FEATURE_COUNT);
  Serial.printf("Nodes: %d\n",
    (int)(sizeof(IF_NODES) / sizeof(IF_NODES[0])));

  Serial.printf("Threshold: %.12f\n", (double)IF_THRESHOLD);

  Serial.println();

  int passing = 0;

  for (int i = 0; i < 5; i++) {
    double score = isolationForestScore(testFeatures[i]);

    bool predictedAnomaly = score < IF_THRESHOLD;
    bool expectedAnomaly = expectedScores[i] < IF_THRESHOLD;

    double difference = fabs(score - expectedScores[i]);

    bool scoreMatches = difference < 0.000001;
    bool classMatches = predictedAnomaly == expectedAnomaly;

    if (scoreMatches && classMatches) {
      passing++;
    }

    Serial.printf("Test: %s\n", testNames[i]);

    Serial.printf("  ESP32 score: %.12f\n", score);
    Serial.printf("  Python score: %.12f\n", expectedScores[i]);
    Serial.printf("  Difference: %.12f\n", difference);

    Serial.printf("  Prediction: %s\n",
      predictedAnomaly ? "ANOMALY" : "NORMAL");

    Serial.printf("  Result: %s\n\n",
      (scoreMatches && classMatches) ? "PASS" : "FAIL");
  }

  Serial.printf("Verification: %d/5 tests passed\n", passing);
  if (passing == 5) {
  Serial.println("MODEL VERIFICATION PASSED");
} else {
  Serial.println("MODEL VERIFICATION FAILED");
}

  const int iterations = 10000;
  volatile double scoreSink = 0.0;

  // Warm up the model
  for (int i = 0; i < 100; i++) {
    scoreSink = isolationForestScore(testFeatures[i % 5]);
  }

  uint32_t startTime = micros();

  for (int i = 0; i < iterations; i++) {
    scoreSink = isolationForestScore(testFeatures[i % 5]);
  }

  uint32_t elapsedTime = micros() - startTime;

  double averageUs =
    (double)elapsedTime / iterations;

  Serial.println();
  Serial.println("ESP32 Inference Benchmark");
  Serial.printf("Iterations: %d\n", iterations);
  Serial.printf("Total time: %lu us\n", (unsigned long)elapsedTime);
  Serial.printf("Average inference time: %.3f us\n", averageUs);
  Serial.printf("Final score: %.9f\n", (double)scoreSink);

  Serial.printf("Free heap: %u bytes\n",
                ESP.getFreeHeap());

  Serial.printf("Sketch size: %u bytes\n",
                ESP.getSketchSize());

  Serial.println();
  Serial.println("Isolation Forest Model Storage");

  Serial.printf("Node structure size: %u bytes\n",
                (unsigned)sizeof(IFNode));

  Serial.printf("Node array size: %u bytes\n",
                (unsigned)sizeof(IF_NODES));

  Serial.printf("Tree offsets size: %u bytes\n",
                (unsigned)sizeof(IF_TREE_OFFSETS));

  Serial.printf("Total exported array storage: %u bytes\n",
                (unsigned)(
                  sizeof(IF_NODES) +
                  sizeof(IF_TREE_OFFSETS)
                ));
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) {
    return;
  }

  float features[IF_FEATURE_COUNT];
  int count = 0;

  char buffer[256];
  line.toCharArray(buffer, sizeof(buffer));

  char* token = strtok(buffer, ",");

  while (token != nullptr && count < IF_FEATURE_COUNT) {
    features[count] = strtof(token, nullptr);
    count++;
    token = strtok(nullptr, ",");
  }

  if (count != IF_FEATURE_COUNT || token != nullptr) {
    Serial.println("ERROR,Expected 9 features");
    return;
  }

  double score = isolationForestScore(features);
  bool anomaly = score < IF_THRESHOLD;

  Serial.printf(
    "RESULT,%.12f,%d\n",
    score,
    anomaly ? 1 : 0
  );
}
