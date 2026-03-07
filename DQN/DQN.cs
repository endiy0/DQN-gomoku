using System.Text.Json;

namespace DQN
{
    public sealed class DQN
    {
        private const int BoardSize = OmokEnvironment.BoardSize;
        private const int BoardArea = BoardSize * BoardSize;
        private const int KernelSize = 3;
        private const int Padding = 1;

        private const int Conv1InChannels = 2;
        private const int Conv1OutChannels = 32;
        private const int Conv2OutChannels = 64;
        private const int Conv3OutChannels = 64;
        private const int Fc1Size = 512;

        private const int Conv3FeatureSize = Conv3OutChannels * BoardArea;

        // Form1.GetTurnAwareObservation이 226개 값을 전달해도 마지막 턴 정보 1칸은 무시하고 보드 225칸만 사용한다.
        public const int InputSize = BoardArea + 1;
        public const int OutputSize = BoardArea;

        private readonly float[] conv1W;
        private readonly float[] conv1B;
        private readonly float[] conv2W;
        private readonly float[] conv2B;
        private readonly float[] conv3W;
        private readonly float[] conv3B;
        private readonly float[] fc1W;
        private readonly float[] fc1B;
        private readonly float[] outW;
        private readonly float[] outB;

        public int HiddenSize => Fc1Size;

        public DQN(int hiddenSize = Fc1Size, int? seed = null)
        {
            conv1W = new float[Conv1OutChannels * Conv1InChannels * KernelSize * KernelSize];
            conv1B = new float[Conv1OutChannels];
            conv2W = new float[Conv2OutChannels * Conv1OutChannels * KernelSize * KernelSize];
            conv2B = new float[Conv2OutChannels];
            conv3W = new float[Conv3OutChannels * Conv2OutChannels * KernelSize * KernelSize];
            conv3B = new float[Conv3OutChannels];
            fc1W = new float[Conv3FeatureSize * Fc1Size];
            fc1B = new float[Fc1Size];
            outW = new float[Fc1Size * OutputSize];
            outB = new float[OutputSize];

            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            InitializeWeights(random);
        }

        private DQN(
            float[] conv1W,
            float[] conv1B,
            float[] conv2W,
            float[] conv2B,
            float[] conv3W,
            float[] conv3B,
            float[] fc1W,
            float[] fc1B,
            float[] outW,
            float[] outB)
        {
            this.conv1W = conv1W;
            this.conv1B = conv1B;
            this.conv2W = conv2W;
            this.conv2B = conv2B;
            this.conv3W = conv3W;
            this.conv3B = conv3B;
            this.fc1W = fc1W;
            this.fc1B = fc1B;
            this.outW = outW;
            this.outB = outB;
        }

        public float[] Predict(float[] state)
        {
            ValidateState(state);

            var x = new float[Conv1InChannels * BoardArea];
            var z1 = new float[Conv1OutChannels * BoardArea];
            var a1 = new float[Conv1OutChannels * BoardArea];
            var z2 = new float[Conv2OutChannels * BoardArea];
            var a2 = new float[Conv2OutChannels * BoardArea];
            var z3 = new float[Conv3OutChannels * BoardArea];
            var a3 = new float[Conv3OutChannels * BoardArea];
            var z4 = new float[Fc1Size];
            var a4 = new float[Fc1Size];
            var q = new float[OutputSize];

            Forward(state, x, z1, a1, z2, a2, z3, a3, z4, a4, q);
            return q;
        }

        public int SelectAction(float[] state, IReadOnlyList<int> legalActions, float epsilon, Random random)
        {
            if (legalActions.Count == 0)
            {
                throw new InvalidOperationException("가능한 행동이 없습니다.");
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
                throw new ArgumentOutOfRangeException(nameof(action), "액션 인덱스가 범위를 벗어났습니다.");
            }

            var x = new float[Conv1InChannels * BoardArea];
            var z1 = new float[Conv1OutChannels * BoardArea];
            var a1 = new float[Conv1OutChannels * BoardArea];
            var z2 = new float[Conv2OutChannels * BoardArea];
            var a2 = new float[Conv2OutChannels * BoardArea];
            var z3 = new float[Conv3OutChannels * BoardArea];
            var a3 = new float[Conv3OutChannels * BoardArea];
            var z4 = new float[Fc1Size];
            var a4 = new float[Fc1Size];
            var q = new float[OutputSize];

            Forward(state, x, z1, a1, z2, a2, z3, a3, z4, a4, q);

            float target = reward;
            if (!done && nextLegalActions.Count > 0)
            {
                float[] nextQ = Predict(nextState);
                float maxNext = float.NegativeInfinity;
                for (int i = 0; i < nextLegalActions.Count; i++)
                {
                    float v = nextQ[nextLegalActions[i]];
                    if (v > maxNext)
                    {
                        maxNext = v;
                    }
                }

                target += gamma * maxNext;
            }

            float tdError = Math.Clamp(q[action] - target, -10f, 10f);

            // Output layer backprop: only selected action.
            var dA4 = new float[Fc1Size];
            for (int j = 0; j < Fc1Size; j++)
            {
                int idx = OutWIndex(j, action);
                float wOld = outW[idx];
                dA4[j] = tdError * wOld;
                outW[idx] -= learningRate * (tdError * a4[j]);
            }
            outB[action] -= learningRate * tdError;

            // FC1 backprop.
            var dZ4 = new float[Fc1Size];
            for (int j = 0; j < Fc1Size; j++)
            {
                dZ4[j] = z4[j] > 0f ? dA4[j] : 0f;
            }

            var dA3 = new float[Conv3FeatureSize];
            for (int i = 0; i < Conv3FeatureSize; i++)
            {
                float a3v = a3[i];
                int row = i * Fc1Size;
                float gradToA3 = 0f;

                for (int j = 0; j < Fc1Size; j++)
                {
                    float dz = dZ4[j];
                    int idx = row + j;
                    float wOld = fc1W[idx];
                    gradToA3 += wOld * dz;
                    fc1W[idx] -= learningRate * (a3v * dz);
                }

                dA3[i] = gradToA3;
            }

            for (int j = 0; j < Fc1Size; j++)
            {
                fc1B[j] -= learningRate * dZ4[j];
            }

            // Conv3 backprop.
            var dZ3 = new float[Conv3OutChannels * BoardArea];
            for (int i = 0; i < dZ3.Length; i++)
            {
                dZ3[i] = z3[i] > 0f ? dA3[i] : 0f;
            }

            var dA2 = new float[Conv2OutChannels * BoardArea];
            BackwardConvSame(
                dZ3,
                a2,
                Conv3OutChannels,
                Conv2OutChannels,
                conv3W,
                conv3B,
                dA2,
                learningRate);

            // Conv2 backprop.
            var dZ2 = new float[Conv2OutChannels * BoardArea];
            for (int i = 0; i < dZ2.Length; i++)
            {
                dZ2[i] = z2[i] > 0f ? dA2[i] : 0f;
            }

            var dA1 = new float[Conv1OutChannels * BoardArea];
            BackwardConvSame(
                dZ2,
                a1,
                Conv2OutChannels,
                Conv1OutChannels,
                conv2W,
                conv2B,
                dA1,
                learningRate);

            // Conv1 backprop.
            var dZ1 = new float[Conv1OutChannels * BoardArea];
            for (int i = 0; i < dZ1.Length; i++)
            {
                dZ1[i] = z1[i] > 0f ? dA1[i] : 0f;
            }

            BackwardConvSame(
                dZ1,
                x,
                Conv1OutChannels,
                Conv1InChannels,
                conv1W,
                conv1B,
                dInput: null,
                learningRate: learningRate);
        }

        public DQN Clone()
        {
            return new DQN(
                (float[])conv1W.Clone(),
                (float[])conv1B.Clone(),
                (float[])conv2W.Clone(),
                (float[])conv2B.Clone(),
                (float[])conv3W.Clone(),
                (float[])conv3B.Clone(),
                (float[])fc1W.Clone(),
                (float[])fc1B.Clone(),
                (float[])outW.Clone(),
                (float[])outB.Clone());
        }

        public void Mutate(float scale, Random random)
        {
            MutateArray(conv1W, scale, random);
            MutateArray(conv1B, scale, random);
            MutateArray(conv2W, scale, random);
            MutateArray(conv2B, scale, random);
            MutateArray(conv3W, scale, random);
            MutateArray(conv3B, scale, random);
            MutateArray(fc1W, scale, random);
            MutateArray(fc1B, scale, random);
            MutateArray(outW, scale, random);
            MutateArray(outB, scale, random);
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
                Version = 4,
                Conv1WB64 = EncodeFloatArray(conv1W),
                Conv1BB64 = EncodeFloatArray(conv1B),
                Conv2WB64 = EncodeFloatArray(conv2W),
                Conv2BB64 = EncodeFloatArray(conv2B),
                Conv3WB64 = EncodeFloatArray(conv3W),
                Conv3BB64 = EncodeFloatArray(conv3B),
                Fc1WB64 = EncodeFloatArray(fc1W),
                Fc1BB64 = EncodeFloatArray(fc1B),
                OutWB64 = EncodeFloatArray(outW),
                OutBB64 = EncodeFloatArray(outB)
            };

            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
            JsonSerializer.Serialize(fs, snapshot);
        }

        public static DQN Load(string path)
        {
            string json = File.ReadAllText(path);
            DqnSnapshot? snapshot = JsonSerializer.Deserialize<DqnSnapshot>(json);
            if (snapshot is null)
            {
                throw new InvalidOperationException("모델 스냅샷을 불러오지 못했습니다.");
            }

            float[] conv1W = ReadTensor(snapshot.Conv1WB64, snapshot.Conv1W, Conv1OutChannels * Conv1InChannels * KernelSize * KernelSize, "Conv1W");
            float[] conv1B = ReadTensor(snapshot.Conv1BB64, snapshot.Conv1B, Conv1OutChannels, "Conv1B");
            float[] conv2W = ReadTensor(snapshot.Conv2WB64, snapshot.Conv2W, Conv2OutChannels * Conv1OutChannels * KernelSize * KernelSize, "Conv2W");
            float[] conv2B = ReadTensor(snapshot.Conv2BB64, snapshot.Conv2B, Conv2OutChannels, "Conv2B");
            float[] conv3W = ReadTensor(snapshot.Conv3WB64, snapshot.Conv3W, Conv3OutChannels * Conv2OutChannels * KernelSize * KernelSize, "Conv3W");
            float[] conv3B = ReadTensor(snapshot.Conv3BB64, snapshot.Conv3B, Conv3OutChannels, "Conv3B");
            float[] fc1W = ReadTensor(snapshot.Fc1WB64, snapshot.Fc1W, Conv3FeatureSize * Fc1Size, "Fc1W");
            float[] fc1B = ReadTensor(snapshot.Fc1BB64, snapshot.Fc1B, Fc1Size, "Fc1B");
            float[] outW = ReadTensor(snapshot.OutWB64, snapshot.OutW, Fc1Size * OutputSize, "OutW");
            float[] outB = ReadTensor(snapshot.OutBB64, snapshot.OutB, OutputSize, "OutB");

            return new DQN(
                (float[])conv1W.Clone(),
                (float[])conv1B.Clone(),
                (float[])conv2W.Clone(),
                (float[])conv2B.Clone(),
                (float[])conv3W.Clone(),
                (float[])conv3B.Clone(),
                (float[])fc1W.Clone(),
                (float[])fc1B.Clone(),
                (float[])outW.Clone(),
                (float[])outB.Clone());
        }

        private void Forward(
            float[] state,
            float[] x,
            float[] z1,
            float[] a1,
            float[] z2,
            float[] a2,
            float[] z3,
            float[] a3,
            float[] z4,
            float[] a4,
            float[] q)
        {
            StateToTwoChannelInput(state, x);

            ConvForwardSame(x, Conv1InChannels, Conv1OutChannels, conv1W, conv1B, z1, a1);
            ConvForwardSame(a1, Conv1OutChannels, Conv2OutChannels, conv2W, conv2B, z2, a2);
            ConvForwardSame(a2, Conv2OutChannels, Conv3OutChannels, conv3W, conv3B, z3, a3);

            for (int j = 0; j < Fc1Size; j++)
            {
                float sum = fc1B[j];
                for (int i = 0; i < Conv3FeatureSize; i++)
                {
                    sum += a3[i] * fc1W[Fc1WIndex(i, j)];
                }

                z4[j] = sum;
                a4[j] = sum > 0f ? sum : 0f;
            }

            for (int action = 0; action < OutputSize; action++)
            {
                float sum = outB[action];
                for (int j = 0; j < Fc1Size; j++)
                {
                    sum += a4[j] * outW[OutWIndex(j, action)];
                }

                q[action] = sum;
            }
        }

        private static void ConvForwardSame(
            float[] input,
            int inChannels,
            int outChannels,
            float[] weights,
            float[] bias,
            float[] zOut,
            float[] aOut)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int y = 0; y < BoardSize; y++)
                {
                    for (int x = 0; x < BoardSize; x++)
                    {
                        float sum = bias[oc];

                        for (int ic = 0; ic < inChannels; ic++)
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

                                    float v = input[FeatureIndex(ic, iy, ix)];
                                    float w = weights[ConvWeightIndex(oc, ic, ky, kx, inChannels)];
                                    sum += v * w;
                                }
                            }
                        }

                        int idx = FeatureIndex(oc, y, x);
                        zOut[idx] = sum;
                        aOut[idx] = sum > 0f ? sum : 0f;
                    }
                }
            }
        }

        private static void BackwardConvSame(
            float[] dZ,
            float[] input,
            int outChannels,
            int inChannels,
            float[] weights,
            float[] bias,
            float[]? dInput,
            float learningRate)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                float biasGrad = 0f;

                for (int y = 0; y < BoardSize; y++)
                {
                    for (int x = 0; x < BoardSize; x++)
                    {
                        float dz = dZ[FeatureIndex(oc, y, x)];
                        if (dz == 0f)
                        {
                            continue;
                        }

                        biasGrad += dz;

                        for (int ic = 0; ic < inChannels; ic++)
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

                                    int wIdx = ConvWeightIndex(oc, ic, ky, kx, inChannels);
                                    float wOld = weights[wIdx];
                                    float inValue = input[FeatureIndex(ic, iy, ix)];

                                    if (dInput is not null)
                                    {
                                        dInput[FeatureIndex(ic, iy, ix)] += dz * wOld;
                                    }

                                    weights[wIdx] -= learningRate * (dz * inValue);
                                }
                            }
                        }
                    }
                }

                bias[oc] -= learningRate * biasGrad;
            }
        }

        private static void StateToTwoChannelInput(float[] state, float[] x)
        {
            for (int idx = 0; idx < BoardArea; idx++)
            {
                float v = state[idx];
                x[FeatureFlatIndex(0, idx)] = v > 0f ? v : 0f;
                x[FeatureFlatIndex(1, idx)] = v < 0f ? -v : 0f;
            }
        }

        private void InitializeWeights(Random random)
        {
            InitHeConv(conv1W, Conv1InChannels, random);
            InitHeConv(conv2W, Conv1OutChannels, random);
            InitHeConv(conv3W, Conv2OutChannels, random);
            InitHeLinear(fc1W, Conv3FeatureSize, random);
            InitHeLinear(outW, Fc1Size, random);
        }

        private static void InitHeConv(float[] weights, int inChannels, Random random)
        {
            float scale = MathF.Sqrt(2f / (inChannels * KernelSize * KernelSize));
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = NextUniform(random, -scale, scale);
            }
        }

        private static void InitHeLinear(float[] weights, int fanIn, Random random)
        {
            float scale = MathF.Sqrt(2f / fanIn);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = NextUniform(random, -scale, scale);
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

        private static void MutateArray(float[] arr, float scale, Random random)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] += NextGaussian(random) * scale;
            }
        }

        private static void ValidateState(float[] state)
        {
            if (state.Length != BoardArea && state.Length != InputSize)
            {
                throw new ArgumentException($"상태 벡터 길이는 {BoardArea} 또는 {InputSize}여야 합니다.", nameof(state));
            }
        }

        private static void ValidateLength(float[] arr, int expected, string name)
        {
            if (arr.Length != expected)
            {
                throw new InvalidOperationException($"{name} 길이가 올바르지 않습니다.");
            }
        }

        private static string EncodeFloatArray(float[] values)
        {
            byte[] bytes = new byte[values.Length * sizeof(float)];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return Convert.ToBase64String(bytes);
        }

        private static float[] ReadTensor(string? encoded, float[]? legacy, int expectedLength, string name)
        {
            if (!string.IsNullOrWhiteSpace(encoded))
            {
                return DecodeFloatArray(encoded, expectedLength, name);
            }

            if (legacy is null)
            {
                throw new InvalidOperationException($"{name} 텐서가 모델 파일에 없습니다.");
            }

            ValidateLength(legacy, expectedLength, name);
            return legacy;
        }

        private static float[] DecodeFloatArray(string encoded, int expectedLength, string name)
        {
            byte[] bytes;
            try
            {
                bytes = Convert.FromBase64String(encoded);
            }
            catch (FormatException ex)
            {
                throw new InvalidOperationException($"{name} 텐서 형식이 올바르지 않습니다.", ex);
            }

            int expectedBytes = expectedLength * sizeof(float);
            if (bytes.Length != expectedBytes)
            {
                throw new InvalidOperationException($"{name} 텐서 크기가 올바르지 않습니다.");
            }

            var values = new float[expectedLength];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        private static int FeatureIndex(int channel, int y, int x)
        {
            return channel * BoardArea + y * BoardSize + x;
        }

        private static int FeatureFlatIndex(int channel, int flatIndex)
        {
            return channel * BoardArea + flatIndex;
        }

        private static int ConvWeightIndex(int outChannel, int inChannel, int ky, int kx, int inChannels)
        {
            return (((outChannel * inChannels + inChannel) * KernelSize + ky) * KernelSize) + kx;
        }

        private static int Fc1WIndex(int featureIndex, int hiddenIndex)
        {
            return featureIndex * Fc1Size + hiddenIndex;
        }

        private static int OutWIndex(int hiddenIndex, int actionIndex)
        {
            return hiddenIndex * OutputSize + actionIndex;
        }

        private sealed class DqnSnapshot
        {
            public int Version { get; set; }
            public string? Conv1WB64 { get; set; }
            public string? Conv1BB64 { get; set; }
            public string? Conv2WB64 { get; set; }
            public string? Conv2BB64 { get; set; }
            public string? Conv3WB64 { get; set; }
            public string? Conv3BB64 { get; set; }
            public string? Fc1WB64 { get; set; }
            public string? Fc1BB64 { get; set; }
            public string? OutWB64 { get; set; }
            public string? OutBB64 { get; set; }
            public float[]? Conv1W { get; set; }
            public float[]? Conv1B { get; set; }
            public float[]? Conv2W { get; set; }
            public float[]? Conv2B { get; set; }
            public float[]? Conv3W { get; set; }
            public float[]? Conv3B { get; set; }
            public float[]? Fc1W { get; set; }
            public float[]? Fc1B { get; set; }
            public float[]? OutW { get; set; }
            public float[]? OutB { get; set; }
        }
    }
}
