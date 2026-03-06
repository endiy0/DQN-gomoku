using System.Text.Json;

namespace DQN
{
    public sealed class DQN
    {
        private const int BoardSize = OmokEnvironment.BoardSize;
        private const int BoardArea = BoardSize * BoardSize;
        private const int InputChannels = 4;
        private const int KernelSize = 3;
        private const int Padding = 1;

        public const int InputSize = BoardArea + 1;
        public const int OutputSize = BoardArea;

        private readonly int filterCount;
        private readonly float[] convW;
        private readonly float[] convB;
        private readonly float[] outW;
        private readonly float[] outB;

        public int HiddenSize => filterCount;

        public DQN(int hiddenSize = 16, int? seed = null)
        {
            filterCount = Math.Max(4, hiddenSize);
            convW = new float[filterCount * InputChannels * KernelSize * KernelSize];
            convB = new float[filterCount];
            outW = new float[filterCount * BoardArea * OutputSize];
            outB = new float[OutputSize];

            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            InitializeWeights(random);
        }

        private DQN(int filterCount, float[] convW, float[] convB, float[] outW, float[] outB)
        {
            this.filterCount = filterCount;
            this.convW = convW;
            this.convB = convB;
            this.outW = outW;
            this.outB = outB;
        }

        public float[] Predict(float[] state)
        {
            ValidateState(state);
            int featureSize = filterCount * BoardArea;
            var tensor = new float[InputChannels * BoardArea];
            var convPre = new float[featureSize];
            var convAct = new float[featureSize];
            var output = new float[OutputSize];

            Forward(state, tensor, convPre, convAct, output);
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

            int featureSize = filterCount * BoardArea;
            var tensor = new float[InputChannels * BoardArea];
            var convPre = new float[featureSize];
            var convAct = new float[featureSize];
            var qValues = new float[OutputSize];

            Forward(state, tensor, convPre, convAct, qValues);

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
            var oldOutColumn = new float[featureSize];

            for (int i = 0; i < featureSize; i++)
            {
                int idx = OutWIndex(i, action);
                oldOutColumn[i] = outW[idx];
                float grad = tdError * convAct[i];
                outW[idx] -= learningRate * grad;
            }

            outB[action] -= learningRate * tdError;

            var convGradW = new float[convW.Length];
            var convGradB = new float[convB.Length];

            for (int f = 0; f < filterCount; f++)
            {
                for (int y = 0; y < BoardSize; y++)
                {
                    for (int x = 0; x < BoardSize; x++)
                    {
                        int featureIndex = FeatureIndex(f, y, x);
                        if (convPre[featureIndex] <= 0f)
                        {
                            continue;
                        }

                        float dPre = tdError * oldOutColumn[featureIndex];
                        if (dPre == 0f)
                        {
                            continue;
                        }

                        convGradB[f] += dPre;

                        for (int c = 0; c < InputChannels; c++)
                        {
                            for (int ky = 0; ky < KernelSize; ky++)
                            {
                                int iy = y + ky - Padding;
                                if (iy < 0 || iy >= BoardSize)
                                {
                                    continue;
                                }

                                for (int kx = 0; kx < KernelSize; kx++)
                                {
                                    int ix = x + kx - Padding;
                                    if (ix < 0 || ix >= BoardSize)
                                    {
                                        continue;
                                    }

                                    float inputValue = tensor[TensorIndex(c, iy, ix)];
                                    int wIdx = ConvWIndex(f, c, ky, kx);
                                    convGradW[wIdx] += dPre * inputValue;
                                }
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < convW.Length; i++)
            {
                convW[i] -= learningRate * convGradW[i];
            }

            for (int i = 0; i < convB.Length; i++)
            {
                convB[i] -= learningRate * convGradB[i];
            }
        }

        public DQN Clone()
        {
            return new DQN(
                filterCount,
                (float[])convW.Clone(),
                (float[])convB.Clone(),
                (float[])outW.Clone(),
                (float[])outB.Clone());
        }

        public void Mutate(float scale, Random random)
        {
            for (int i = 0; i < convW.Length; i++)
            {
                convW[i] += NextGaussian(random) * scale;
            }

            for (int i = 0; i < convB.Length; i++)
            {
                convB[i] += NextGaussian(random) * scale;
            }

            for (int i = 0; i < outW.Length; i++)
            {
                outW[i] += NextGaussian(random) * scale;
            }

            for (int i = 0; i < outB.Length; i++)
            {
                outB[i] += NextGaussian(random) * scale;
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
                Version = 2,
                FilterCount = filterCount,
                ConvW = convW,
                ConvB = convB,
                OutW = outW,
                OutB = outB
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

            if (snapshot.ConvW is null || snapshot.ConvB is null || snapshot.OutW is null || snapshot.OutB is null)
            {
                throw new InvalidOperationException("CNN DQN 형식이 아닌 모델 파일입니다.");
            }

            int filters = snapshot.FilterCount;
            int expectedConvW = filters * InputChannels * KernelSize * KernelSize;
            int expectedOutW = filters * BoardArea * OutputSize;

            if (snapshot.ConvW.Length != expectedConvW)
            {
                throw new InvalidOperationException("ConvW 크기가 올바르지 않습니다.");
            }

            if (snapshot.ConvB.Length != filters)
            {
                throw new InvalidOperationException("ConvB 크기가 올바르지 않습니다.");
            }

            if (snapshot.OutW.Length != expectedOutW)
            {
                throw new InvalidOperationException("OutW 크기가 올바르지 않습니다.");
            }

            if (snapshot.OutB.Length != OutputSize)
            {
                throw new InvalidOperationException("OutB 크기가 올바르지 않습니다.");
            }

            return new DQN(
                filters,
                (float[])snapshot.ConvW.Clone(),
                (float[])snapshot.ConvB.Clone(),
                (float[])snapshot.OutW.Clone(),
                (float[])snapshot.OutB.Clone());
        }

        private void Forward(float[] state, float[] tensor, float[] convPre, float[] convAct, float[] output)
        {
            StateToTensor(state, tensor);

            for (int f = 0; f < filterCount; f++)
            {
                for (int y = 0; y < BoardSize; y++)
                {
                    for (int x = 0; x < BoardSize; x++)
                    {
                        float sum = convB[f];

                        for (int c = 0; c < InputChannels; c++)
                        {
                            for (int ky = 0; ky < KernelSize; ky++)
                            {
                                int iy = y + ky - Padding;
                                if (iy < 0 || iy >= BoardSize)
                                {
                                    continue;
                                }

                                for (int kx = 0; kx < KernelSize; kx++)
                                {
                                    int ix = x + kx - Padding;
                                    if (ix < 0 || ix >= BoardSize)
                                    {
                                        continue;
                                    }

                                    float inputValue = tensor[TensorIndex(c, iy, ix)];
                                    float weight = convW[ConvWIndex(f, c, ky, kx)];
                                    sum += inputValue * weight;
                                }
                            }
                        }

                        int idx = FeatureIndex(f, y, x);
                        convPre[idx] = sum;
                        convAct[idx] = sum > 0f ? sum : 0f;
                    }
                }
            }

            Array.Copy(outB, output, OutputSize);

            int featureSize = filterCount * BoardArea;
            for (int i = 0; i < featureSize; i++)
            {
                float activation = convAct[i];
                if (activation == 0f)
                {
                    continue;
                }

                int rowOffset = i * OutputSize;
                for (int action = 0; action < OutputSize; action++)
                {
                    output[action] += activation * outW[rowOffset + action];
                }
            }
        }

        private static void StateToTensor(float[] state, float[] tensor)
        {
            float turn = state[BoardArea];
            float turnPlaneValue = (turn + 1f) * 0.5f;

            for (int idx = 0; idx < BoardArea; idx++)
            {
                float v = state[idx];
                float own = v > 0f ? v : 0f;
                float opp = v < 0f ? -v : 0f;
                float empty = 1f - Math.Clamp(MathF.Abs(v), 0f, 1f);

                tensor[TensorIndex(0, idx)] = own;
                tensor[TensorIndex(1, idx)] = opp;
                tensor[TensorIndex(2, idx)] = empty;
                tensor[TensorIndex(3, idx)] = turnPlaneValue;
            }
        }

        private void InitializeWeights(Random random)
        {
            float convScale = MathF.Sqrt(2f / (InputChannels * KernelSize * KernelSize));
            float outScale = MathF.Sqrt(2f / (filterCount * BoardArea));

            for (int i = 0; i < convW.Length; i++)
            {
                convW[i] = NextUniform(random, -convScale, convScale);
            }

            for (int i = 0; i < outW.Length; i++)
            {
                outW[i] = NextUniform(random, -outScale, outScale);
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

        private static void ValidateState(float[] state)
        {
            if (state.Length != InputSize)
            {
                throw new ArgumentException($"상태 벡터 길이는 {InputSize}이어야 합니다.", nameof(state));
            }
        }

        private static int TensorIndex(int channel, int y, int x)
        {
            return (channel * BoardArea) + (y * BoardSize + x);
        }

        private static int TensorIndex(int channel, int flatIndex)
        {
            return channel * BoardArea + flatIndex;
        }

        private static int FeatureIndex(int filter, int y, int x)
        {
            return (filter * BoardArea) + (y * BoardSize + x);
        }

        private static int ConvWIndex(int filter, int channel, int ky, int kx)
        {
            return (((filter * InputChannels + channel) * KernelSize + ky) * KernelSize) + kx;
        }

        private static int OutWIndex(int featureIndex, int action)
        {
            return featureIndex * OutputSize + action;
        }

        private sealed class DqnSnapshot
        {
            public int Version { get; set; }
            public int FilterCount { get; set; }
            public float[]? ConvW { get; set; }
            public float[]? ConvB { get; set; }
            public float[]? OutW { get; set; }
            public float[]? OutB { get; set; }
        }
    }
}
