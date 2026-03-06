using System.Text.Json;

namespace DQN
{
    public sealed class DQN
    {
        public const int InputSize = OmokEnvironment.ActionCount + 1;
        public const int OutputSize = OmokEnvironment.ActionCount;

        private readonly int hiddenSize;
        private readonly float[] w1;
        private readonly float[] b1;
        private readonly float[] w2;
        private readonly float[] b2;

        public int HiddenSize => hiddenSize;

        public DQN(int hiddenSize = 128, int? seed = null)
        {
            this.hiddenSize = hiddenSize;

            w1 = new float[InputSize * hiddenSize];
            b1 = new float[hiddenSize];
            w2 = new float[hiddenSize * OutputSize];
            b2 = new float[OutputSize];

            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            InitializeWeights(random);
        }

        private DQN(int hiddenSize, float[] w1, float[] b1, float[] w2, float[] b2)
        {
            this.hiddenSize = hiddenSize;
            this.w1 = w1;
            this.b1 = b1;
            this.w2 = w2;
            this.b2 = b2;
        }

        public float[] Predict(float[] state)
        {
            ValidateState(state);
            var hiddenPre = new float[hiddenSize];
            var hiddenAct = new float[hiddenSize];
            var output = new float[OutputSize];
            Forward(state, hiddenPre, hiddenAct, output);
            return output;
        }

        public int SelectAction(float[] state, IReadOnlyList<int> legalActions, float epsilon, Random random)
        {
            if (legalActions.Count == 0)
            {
                throw new InvalidOperationException("합법 액션이 없습니다.");
            }

            if (random.NextDouble() < epsilon)
            {
                return legalActions[random.Next(legalActions.Count)];
            }

            float[] qValues = Predict(state);
            int bestAction = legalActions[0];
            float bestValue = qValues[bestAction];

            for (int i = 1; i < legalActions.Count; i++)
            {
                int action = legalActions[i];
                float value = qValues[action];
                if (value > bestValue)
                {
                    bestValue = value;
                    bestAction = action;
                }
            }

            return bestAction;
        }

        public void Train(
            float[] state,
            int action,
            float reward,
            float[] nextState,
            IReadOnlyList<int> nextLegalActions,
            bool done,
            float gamma,
            float learningRate)
        {
            ValidateState(state);
            ValidateState(nextState);

            if (action < 0 || action >= OutputSize)
            {
                throw new ArgumentOutOfRangeException(nameof(action), "유효하지 않은 액션 인덱스입니다.");
            }

            var hiddenPre = new float[hiddenSize];
            var hiddenAct = new float[hiddenSize];
            var qValues = new float[OutputSize];
            Forward(state, hiddenPre, hiddenAct, qValues);

            float target = reward;
            if (!done && nextLegalActions.Count > 0)
            {
                float[] nextQ = Predict(nextState);
                float maxNext = float.NegativeInfinity;
                for (int i = 0; i < nextLegalActions.Count; i++)
                {
                    float q = nextQ[nextLegalActions[i]];
                    if (q > maxNext)
                    {
                        maxNext = q;
                    }
                }

                target += gamma * maxNext;
            }

            float tdError = Math.Clamp(qValues[action] - target, -10f, 10f);
            var oldW2Column = new float[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                oldW2Column[h] = w2[W2Index(h, action)];
            }

            for (int h = 0; h < hiddenSize; h++)
            {
                int idx = W2Index(h, action);
                float grad = tdError * hiddenAct[h];
                w2[idx] -= learningRate * grad;
            }
            b2[action] -= learningRate * tdError;

            for (int h = 0; h < hiddenSize; h++)
            {
                if (hiddenPre[h] <= 0f)
                {
                    continue;
                }

                float hiddenGrad = tdError * oldW2Column[h];
                int w1ColOffset = h;

                for (int i = 0; i < InputSize; i++)
                {
                    int idx = W1Index(i, w1ColOffset);
                    float grad = hiddenGrad * state[i];
                    w1[idx] -= learningRate * grad;
                }

                b1[h] -= learningRate * hiddenGrad;
            }
        }

        public DQN Clone()
        {
            return new DQN(
                hiddenSize,
                (float[])w1.Clone(),
                (float[])b1.Clone(),
                (float[])w2.Clone(),
                (float[])b2.Clone());
        }

        public void Mutate(float scale, Random random)
        {
            for (int i = 0; i < w1.Length; i++)
            {
                w1[i] += NextGaussian(random) * scale;
            }

            for (int i = 0; i < b1.Length; i++)
            {
                b1[i] += NextGaussian(random) * scale;
            }

            for (int i = 0; i < w2.Length; i++)
            {
                w2[i] += NextGaussian(random) * scale;
            }

            for (int i = 0; i < b2.Length; i++)
            {
                b2[i] += NextGaussian(random) * scale;
            }
        }

        public void Save(string path)
        {
            string? directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var snapshot = new DqnSnapshot
            {
                HiddenSize = hiddenSize,
                W1 = w1,
                B1 = b1,
                W2 = w2,
                B2 = b2
            };

            string json = JsonSerializer.Serialize(snapshot, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, json);
        }

        public static DQN Load(string path)
        {
            string json = File.ReadAllText(path);
            DqnSnapshot? snapshot = JsonSerializer.Deserialize<DqnSnapshot>(json);
            if (snapshot is null)
            {
                throw new InvalidOperationException("모델 파일을 읽을 수 없습니다.");
            }

            int hidden = snapshot.HiddenSize;
            if (snapshot.W1 is null || snapshot.B1 is null || snapshot.W2 is null || snapshot.B2 is null)
            {
                throw new InvalidOperationException("모델 파일 형식이 잘못되었습니다.");
            }

            if (snapshot.W1.Length != InputSize * hidden)
            {
                throw new InvalidOperationException("W1 크기가 올바르지 않습니다.");
            }

            if (snapshot.B1.Length != hidden)
            {
                throw new InvalidOperationException("B1 크기가 올바르지 않습니다.");
            }

            if (snapshot.W2.Length != hidden * OutputSize)
            {
                throw new InvalidOperationException("W2 크기가 올바르지 않습니다.");
            }

            if (snapshot.B2.Length != OutputSize)
            {
                throw new InvalidOperationException("B2 크기가 올바르지 않습니다.");
            }

            return new DQN(
                hidden,
                (float[])snapshot.W1.Clone(),
                (float[])snapshot.B1.Clone(),
                (float[])snapshot.W2.Clone(),
                (float[])snapshot.B2.Clone());
        }

        private void InitializeWeights(Random random)
        {
            float w1Scale = MathF.Sqrt(2f / InputSize);
            float w2Scale = MathF.Sqrt(2f / hiddenSize);

            for (int i = 0; i < w1.Length; i++)
            {
                w1[i] = NextUniform(random, -w1Scale, w1Scale);
            }

            for (int i = 0; i < w2.Length; i++)
            {
                w2[i] = NextUniform(random, -w2Scale, w2Scale);
            }
        }

        private static float NextUniform(Random random, float min, float max)
        {
            return min + (float)random.NextDouble() * (max - min);
        }

        private static float NextGaussian(Random random)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double stdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return (float)stdNormal;
        }

        private void Forward(float[] input, float[] hiddenPre, float[] hiddenAct, float[] output)
        {
            for (int h = 0; h < hiddenSize; h++)
            {
                float sum = b1[h];
                for (int i = 0; i < InputSize; i++)
                {
                    sum += input[i] * w1[W1Index(i, h)];
                }

                hiddenPre[h] = sum;
                hiddenAct[h] = sum > 0f ? sum : 0f;
            }

            for (int o = 0; o < OutputSize; o++)
            {
                float sum = b2[o];
                for (int h = 0; h < hiddenSize; h++)
                {
                    sum += hiddenAct[h] * w2[W2Index(h, o)];
                }

                output[o] = sum;
            }
        }

        private static void ValidateState(float[] state)
        {
            if (state.Length != InputSize)
            {
                throw new ArgumentException($"상태 벡터 길이는 {InputSize}이어야 합니다.", nameof(state));
            }
        }

        private int W1Index(int input, int hidden)
        {
            return input * hiddenSize + hidden;
        }

        private int W2Index(int hidden, int output)
        {
            return hidden * OutputSize + output;
        }

        private sealed class DqnSnapshot
        {
            public int HiddenSize { get; set; }
            public float[]? W1 { get; set; }
            public float[]? B1 { get; set; }
            public float[]? W2 { get; set; }
            public float[]? B2 { get; set; }
        }
    }
}
