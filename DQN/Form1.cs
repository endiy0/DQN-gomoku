namespace DQN
{
    public partial class Form1 : Form
    {
        private const int BoardSize = OmokEnvironment.BoardSize;
        private const int CellSize = 34;
        private const int BoardLeft = 40;
        private const int BoardTop = 90;
        private const int BoardRight = 40;
        private const int BoardBottom = 40;
        private const int StoneDiameter = 26;
        private const int StarPointDiameter = 6;

        private readonly OmokEnvironment environment = new();
        private readonly Label statusLabel = new();
        private readonly object modelLock = new();
        private readonly object saveQueueLock = new();
        private readonly Random random = new();
        private readonly string modelDirectory = Path.Combine(AppContext.BaseDirectory, "models");
        private readonly string latestModelPath;

        private CancellationTokenSource? trainingCts;
        private Task? trainingTask;
        private Task? saveWorkerTask;
        private DQN? championModel;
        private DQN? queuedSaveModel;
        private bool saveWorkerRunning;
        private bool isTraining;
        private bool aiThinking;
        private StoneColor[,]? trainingRenderBoard;
        private int? trainingRenderLastAction;

        private PlayMode playMode = PlayMode.LocalTwoPlayer;
        private readonly List<GameExperience> gameExperiences = [];

        public OmokEnvironment Environment => environment;

        private enum PlayMode
        {
            LocalTwoPlayer,
            PlayerVsAi,
            Training
        }

        private readonly record struct TrainingResult(StoneColor Winner, int MoveCount);
        private readonly record struct GameExperience(float[] State, int ActionIndex, StoneColor Player);
        private readonly record struct TrainingMove(float[] State, int ActionIndex, StoneColor Player);

        public Form1()
        {
            latestModelPath = Path.Combine(modelDirectory, "latest_winner.json");

            InitializeComponent();

            ClientSize = new Size(
                BoardLeft + BoardRight + CellSize * (BoardSize - 1),
                BoardTop + BoardBottom + CellSize * (BoardSize - 1));
            BackColor = Color.BurlyWood;
            Text = "오목 DQN";
            DoubleBuffered = true;
            KeyPreview = true;

            ConfigureHud();
            TryLoadLatestModel();
            StartNewGame();

            Paint += Form1_Paint;
            MouseClick += Form1_MouseClick;
            KeyDown += Form1_KeyDown;
        }

        private void ConfigureHud()
        {
            statusLabel.AutoSize = true;
            statusLabel.Font = new Font("Malgun Gothic", 10, FontStyle.Bold);
            statusLabel.Location = new Point(230, 24);
            statusLabel.BackColor = Color.Transparent;
            Controls.Add(statusLabel);

            button1.Text = "학습";
            button2.Text = "대국";
            button1.Click += TrainButton_Click;
            button2.Click += GameButton_Click;
        }

        private void TryLoadLatestModel()
        {
            try
            {
                if (!File.Exists(latestModelPath))
                {
                    return;
                }

                championModel = DQN.Load(latestModelPath);
                UpdateStatus("디스크에서 최신 모델을 불러왔습니다.");
            }
            catch (Exception ex)
            {
                UpdateStatus($"모델을 불러오지 못했습니다: {ex.Message}");
            }
        }

        private void StartNewGame()
        {
            environment.Reset();
            aiThinking = false;
            gameExperiences.Clear();
            if (playMode != PlayMode.Training)
            {
                ClearTrainingRender();
            }
            UpdateStatusByMode();
            Invalidate();
        }

        private void UndoLastMove()
        {
            if (playMode == PlayMode.Training)
            {
                return;
            }

            if (playMode == PlayMode.PlayerVsAi)
            {
                UpdateStatus("플레이어 대 AI 모드에서는 무르기를 사용할 수 없습니다.");
                return;
            }

            if (!environment.UndoLastAction())
            {
                UpdateStatus("무를 수 있는 수가 없습니다.");
                return;
            }

            UpdateStatusByMode();
            Invalidate();
        }

        private void UpdateStatus(string message)
        {
            statusLabel.Text = $"{message}  (R: 다시 시작, Backspace: 무르기)";
        }

        private void UpdateStatusByMode()
        {
            string modeText = playMode switch
            {
                PlayMode.LocalTwoPlayer => "로컬 2인 대전",
                PlayMode.PlayerVsAi => "대국 모드 (플레이어: 흑 / AI: 백)",
                PlayMode.Training => "학습 모드",
                _ => "알 수 없는 모드"
            };

            UpdateStatus($"{modeText} - {environment.GetStatusText()}");
        }

        private void SetStatusThreadSafe(string message)
        {
            if (IsDisposed || Disposing)
            {
                return;
            }

            if (InvokeRequired)
            {
                BeginInvoke(new Action<string>(SetStatusThreadSafe), message);
                return;
            }

            UpdateStatus(message);
        }

        private void Form1_Paint(object? sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            DrawBoard(g);
            DrawStones(g);
        }

        private void DrawBoard(Graphics g)
        {
            int startX = BoardLeft;
            int startY = BoardTop;
            int endX = BoardLeft + CellSize * (BoardSize - 1);
            int endY = BoardTop + CellSize * (BoardSize - 1);

            using var gridPen = new Pen(Color.Black, 1);

            for (int i = 0; i < BoardSize; i++)
            {
                int x = BoardLeft + i * CellSize;
                int y = BoardTop + i * CellSize;
                g.DrawLine(gridPen, startX, y, endX, y);
                g.DrawLine(gridPen, x, startY, x, endY);
            }

            DrawStarPoints(g);
        }

        private void DrawStarPoints(Graphics g)
        {
            int[] indexes = [3, 7, 11];
            int radius = StarPointDiameter / 2;

            foreach (int y in indexes)
            {
                foreach (int x in indexes)
                {
                    int centerX = BoardLeft + x * CellSize;
                    int centerY = BoardTop + y * CellSize;
                    var rect = new Rectangle(centerX - radius, centerY - radius, StarPointDiameter, StarPointDiameter);
                    g.FillEllipse(Brushes.Black, rect);
                }
            }
        }

        private void DrawStones(Graphics g)
        {
            int radius = StoneDiameter / 2;
            StoneColor[,]? boardForRender = playMode == PlayMode.Training ? trainingRenderBoard : null;
            int? lastActionForRender = playMode == PlayMode.Training ? trainingRenderLastAction : environment.LastActionIndex;

            Point? lastMove = lastActionForRender.HasValue
                ? ToPoint(OmokEnvironment.ToBoardPosition(lastActionForRender.Value))
                : null;

            for (int y = 0; y < BoardSize; y++)
            {
                for (int x = 0; x < BoardSize; x++)
                {
                    StoneColor stone = boardForRender is null ? environment.GetStoneAt(x, y) : boardForRender[x, y];
                    if (stone == StoneColor.None)
                    {
                        continue;
                    }

                    int centerX = BoardLeft + x * CellSize;
                    int centerY = BoardTop + y * CellSize;
                    var rect = new Rectangle(centerX - radius, centerY - radius, StoneDiameter, StoneDiameter);

                    if (stone == StoneColor.Black)
                    {
                        g.FillEllipse(Brushes.Black, rect);
                    }
                    else
                    {
                        g.FillEllipse(Brushes.White, rect);
                        g.DrawEllipse(Pens.Black, rect);
                    }

                    if (lastMove.HasValue && lastMove.Value.X == x && lastMove.Value.Y == y)
                    {
                        using var markerPen = new Pen(Color.Red, 2);
                        g.DrawEllipse(markerPen, rect);
                    }
                }
            }
        }

        private void Form1_MouseClick(object? sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
            {
                return;
            }

            if (playMode == PlayMode.Training)
            {
                UpdateStatus("학습 중에는 돌을 놓을 수 없습니다.");
                return;
            }

            if (aiThinking)
            {
                UpdateStatus("AI가 수를 계산 중입니다.");
                return;
            }

            if (!TryGetBoardPosition(e.Location, out int x, out int y))
            {
                return;
            }

            if (playMode == PlayMode.PlayerVsAi && environment.CurrentTurn != StoneColor.Black)
            {
                UpdateStatus("플레이어는 흑돌 차례에만 둘 수 있습니다.");
                return;
            }

            float[]? playerState = null;
            if (playMode == PlayMode.PlayerVsAi)
            {
                playerState = environment.GetTurnAwareObservation();
            }

            if (!environment.TryStepAt(x, y, out OmokStepResult step))
            {
                UpdateStatus(step.Message);
                return;
            }

            if (playMode == PlayMode.PlayerVsAi && playerState is not null)
            {
                gameExperiences.Add(new GameExperience((float[])playerState.Clone(), step.ActionIndex, StoneColor.Black));
            }

            HandleStepResult(step, showDialog: true);

            if (playMode == PlayMode.PlayerVsAi && !step.IsGameOver && environment.CurrentTurn == StoneColor.White)
            {
                _ = RunAiTurnAsync();
            }
        }

        private void HandleStepResult(OmokStepResult step, bool showDialog)
        {
            if (!step.IsGameOver)
            {
                UpdateStatusByMode();
                Invalidate();
                return;
            }

            if (playMode == PlayMode.PlayerVsAi)
            {
                TrainChampionFromPlayerGame(step.Winner);
            }

            if (step.Winner == StoneColor.None)
            {
                UpdateStatus("무승부입니다. R 키로 새 게임을 시작하세요.");
                Invalidate();
                if (showDialog)
                {
                    MessageBox.Show("무승부입니다.", "대국 결과", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                return;
            }

            string winner = OmokEnvironment.GetStoneName(step.Winner);
            UpdateStatus($"{winner} 승리! R 키로 새 게임을 시작하세요.");
            Invalidate();
            if (showDialog)
            {
                MessageBox.Show($"{winner} 승리입니다.", "대국 결과", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        private async Task RunAiTurnAsync()
        {
            if (aiThinking || playMode != PlayMode.PlayerVsAi || environment.IsGameOver || environment.CurrentTurn != StoneColor.White)
            {
                return;
            }

            aiThinking = true;
            try
            {
                await Task.Delay(120);

                if (playMode != PlayMode.PlayerVsAi || environment.IsGameOver || environment.CurrentTurn != StoneColor.White)
                {
                    return;
                }

                List<int> legalActions = environment.GetLegalActions();
                if (legalActions.Count == 0)
                {
                    return;
                }

                DQN? model = GetChampionSnapshot();
                float[] aiState = environment.GetTurnAwareObservation();
                int action = model is null
                    ? legalActions[random.Next(legalActions.Count)]
                    : model.SelectAction(aiState, legalActions, epsilon: 0f, random);

                gameExperiences.Add(new GameExperience((float[])aiState.Clone(), action, StoneColor.White));

                OmokStepResult aiStep = environment.StepByAction(action);
                HandleStepResult(aiStep, showDialog: true);
            }
            finally
            {
                aiThinking = false;
            }
        }

        private DQN? GetChampionSnapshot()
        {
            lock (modelLock)
            {
                return championModel?.Clone();
            }
        }

        private async void TrainButton_Click(object? sender, EventArgs e)
        {
            if (isTraining)
            {
                await StopTrainingAsync();
                return;
            }

            await StartTrainingAsync();
        }

        private async Task StartTrainingAsync()
        {
            if (isTraining)
            {
                return;
            }

            playMode = PlayMode.Training;
            isTraining = true;
            button1.Text = "중지";
            button2.Enabled = false;
            ClearTrainingRender();

            trainingCts = new CancellationTokenSource();
            trainingTask = Task.Run(() => RunTrainingLoopAsync(trainingCts.Token));

            UpdateStatus("학습 시작: 두 모델이 self-play를 진행합니다.");
            await Task.CompletedTask;
        }

        private async Task StopTrainingAsync()
        {
            if (!isTraining)
            {
                return;
            }

            trainingCts?.Cancel();

            if (trainingTask is not null)
            {
                try
                {
                    await trainingTask;
                }
                catch (OperationCanceledException)
                {
                }
                catch (Exception ex)
                {
                    UpdateStatus($"학습 중 오류가 발생했습니다: {ex.Message}");
                }
            }

            trainingCts?.Dispose();
            trainingCts = null;
            trainingTask = null;
            isTraining = false;

            button1.Text = "학습";
            button2.Enabled = true;
            playMode = PlayMode.LocalTwoPlayer;
            ClearTrainingRender();
            Invalidate();
            UpdateStatus("학습을 중지했습니다. 대국 버튼으로 플레이어 대 AI 모드를 시작할 수 있습니다.");
        }

        private async Task RunTrainingLoopAsync(CancellationToken token)
        {
            Directory.CreateDirectory(modelDirectory);
            gameExperiences.Clear();
            long lastFrameTick = System.Environment.TickCount64;

            var loopRandom = new Random(Guid.NewGuid().GetHashCode());
            DQN? seedModel = GetChampionSnapshot();
            DQN modelA = seedModel?.Clone() ?? new DQN(seed: loopRandom.Next());
            DQN modelB = seedModel?.Clone() ?? new DQN(seed: loopRandom.Next());
            modelB.Mutate(scale: 0.03f, loopRandom);

            int gameCount = 0;
            int blackWins = 0;
            int whiteWins = 0;
            int draws = 0;

            while (!token.IsCancellationRequested)
            {
                gameCount++;

                TrainingResult result;
                try
                {
                    result = PlayTrainingGame(modelA, modelB, gameCount, loopRandom, token, ref lastFrameTick);
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    SetStatusThreadSafe($"학습 루프 오류: {ex.Message}");
                    await Task.Delay(50, token);
                    continue;
                }

                switch (result.Winner)
                {
                    case StoneColor.Black:
                        blackWins++;
                        break;
                    case StoneColor.White:
                        whiteWins++;
                        break;
                    default:
                        draws++;
                        break;
                }

                DQN winnerModel = PickWinnerModel(modelA, modelB, result.Winner, loopRandom);
                if (gameCount == 1 || gameCount % 10 == 0)
                {
                    EnqueueLatestModelSave(winnerModel);
                }

                lock (modelLock)
                {
                    championModel = winnerModel.Clone();
                }

                modelA = winnerModel.Clone();
                modelB = winnerModel.Clone();
                modelB.Mutate(scale: 0.02f, loopRandom);

                if (gameCount % 2 == 0)
                {
                    (modelA, modelB) = (modelB, modelA);
                }

                string winnerText = result.Winner switch
                {
                    StoneColor.Black => "흑 승",
                    StoneColor.White => "백 승",
                    _ => "무승부(무작위 선택)"
                };

                SetStatusThreadSafe(
                    $"학습 중... {gameCount}판 완료, 우승: {winnerText}, 수: {result.MoveCount}, " +
                    $"전적(B/W/D): {blackWins}/{whiteWins}/{draws}");
                await Task.Yield();
            }
        }

        private static DQN PickWinnerModel(DQN blackModel, DQN whiteModel, StoneColor winner, Random random)
        {
            return winner switch
            {
                StoneColor.Black => blackModel,
                StoneColor.White => whiteModel,
                _ => random.Next(2) == 0 ? blackModel : whiteModel
            };
        }

        private TrainingResult PlayTrainingGame(DQN blackModel, DQN whiteModel, int gameCount, Random loopRandom, CancellationToken token, ref long lastFrameTick)
        {
            var env = new OmokEnvironment();
            env.Reset();
            PublishTrainingFrame(env.GetBoardCopy(), env.LastActionIndex);

            float epsilon = MathF.Max(0.05f, 0.35f - gameCount * 0.02f); //0.0002f
            int moveCount = 0;
            var episodeMoves = new List<TrainingMove>(OmokEnvironment.ActionCount);

            while (!env.IsGameOver)
            {
                token.ThrowIfCancellationRequested();

                StoneColor turn = env.CurrentTurn;
                DQN activeModel = turn == StoneColor.Black ? blackModel : whiteModel;

                float[] state = env.GetTurnAwareObservation();
                List<int> legal = env.GetLegalActions();
                if (legal.Count == 0)
                {
                    break;
                }

                int action = activeModel.SelectAction(state, legal, epsilon, loopRandom);
                episodeMoves.Add(new TrainingMove((float[])state.Clone(), action, turn));
                OmokStepResult step = env.StepByAction(action);

                float reward = step.Reward;
                if (step.IsGameOver && step.Winner == StoneColor.None)
                {
                    reward = 0.2f;
                }

                float[] nextState = env.GetTurnAwareObservation();
                List<int> nextLegal = env.GetLegalActions();

                activeModel.Train(
                    state,
                    action,
                    reward,
                    nextState,
                    nextLegal,
                    step.IsGameOver,
                    gamma: 0.99f,
                    learningRate: 0.0007f);

                moveCount++;
                long now = System.Environment.TickCount64;
                if (now - lastFrameTick >= 150 || step.IsGameOver)
                {
                    lastFrameTick = now;
                    PublishTrainingFrame(env.GetBoardCopy(), env.LastActionIndex);
                }

                if (moveCount >= OmokEnvironment.ActionCount)
                {
                    break;
                }
            }

            ApplyEpisodeResultTraining(blackModel, episodeMoves, StoneColor.Black, env.Winner);
            ApplyEpisodeResultTraining(whiteModel, episodeMoves, StoneColor.White, env.Winner);
            PublishTrainingFrame(env.GetBoardCopy(), env.LastActionIndex);
            return new TrainingResult(env.Winner, moveCount);
        }

        private static void ApplyEpisodeResultTraining(DQN model, List<TrainingMove> moves, StoneColor learner, StoneColor winner)
        {
            float finalReward = winner switch
            {
                StoneColor.None => 0.2f,
                _ when winner == learner => 1f,
                _ => -1f
            };

            int ownMoveIndex = 0;
            for (int i = moves.Count - 1; i >= 0; i--)
            {
                TrainingMove move = moves[i];
                if (move.Player != learner)
                {
                    continue;
                }

                float reward = finalReward * MathF.Pow(0.985f, ownMoveIndex);
                model.Train(
                    move.State,
                    move.ActionIndex,
                    reward,
                    move.State,
                    Array.Empty<int>(),
                    done: true,
                    gamma: 0f,
                    learningRate: 0.0012f);
                ownMoveIndex++;
            }
        }

        private void SaveLatestModel(DQN model)
        {
            Directory.CreateDirectory(modelDirectory);
            model.Save(latestModelPath);
        }

        private void EnqueueLatestModelSave(DQN model)
        {
            lock (saveQueueLock)
            {
                queuedSaveModel = model.Clone();
                if (saveWorkerRunning)
                {
                    return;
                }

                saveWorkerRunning = true;
                saveWorkerTask = Task.Run(ProcessSaveQueueAsync);
            }
        }

        private async Task ProcessSaveQueueAsync()
        {
            while (true)
            {
                DQN? modelToSave;
                lock (saveQueueLock)
                {
                    modelToSave = queuedSaveModel;
                    queuedSaveModel = null;

                    if (modelToSave is null)
                    {
                        saveWorkerRunning = false;
                        return;
                    }
                }

                try
                {
                    SaveLatestModel(modelToSave);
                }
                catch (Exception ex)
                {
                    SetStatusThreadSafe($"모델 저장 실패: {ex.Message}");
                }

                await Task.Yield();
            }
        }

        private void TrainChampionFromPlayerGame(StoneColor winner)
        {
            if (gameExperiences.Count == 0)
            {
                return;
            }

            try
            {
                DQN? modelToSave = null;
                lock (modelLock)
                {
                    championModel ??= new DQN(seed: random.Next());
                    int total = gameExperiences.Count;

                    for (int i = 0; i < total; i++)
                    {
                        GameExperience exp = gameExperiences[i];
                        int remaining = total - 1 - i;
                        float terminalReward = winner switch
                        {
                            StoneColor.None => 0.2f,
                            _ when winner == exp.Player => 1f,
                            _ => -1f
                        };
                        float reward = terminalReward * MathF.Pow(0.98f, remaining);

                        championModel.Train(
                            exp.State,
                            exp.ActionIndex,
                            reward,
                            exp.State,
                            Array.Empty<int>(),
                            done: true,
                            gamma: 0f,
                            learningRate: 0.001f);
                    }

                    modelToSave = championModel.Clone();
                }

                if (modelToSave is not null)
                {
                    EnqueueLatestModelSave(modelToSave);
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"플레이어 대국 학습 오류: {ex.Message}");
            }
            finally
            {
                gameExperiences.Clear();
            }
        }

        private void PublishTrainingFrame(StoneColor[,] boardCopy, int? lastAction)
        {
            if (IsDisposed || Disposing)
            {
                return;
            }

            if (InvokeRequired)
            {
                try
                {
                    BeginInvoke(new Action<StoneColor[,], int?>(PublishTrainingFrameOnUi), boardCopy, lastAction);
                }
                catch
                {
                }

                return;
            }

            PublishTrainingFrameOnUi(boardCopy, lastAction);
        }

        private void PublishTrainingFrameOnUi(StoneColor[,] boardCopy, int? lastAction)
        {
            trainingRenderBoard = boardCopy;
            trainingRenderLastAction = lastAction;
            Invalidate();
        }

        private void ClearTrainingRender()
        {
            trainingRenderBoard = null;
            trainingRenderLastAction = null;
        }

        private async void GameButton_Click(object? sender, EventArgs e)
        {
            if (isTraining)
            {
                await StopTrainingAsync();
            }

            playMode = PlayMode.PlayerVsAi;
            StartNewGame();
            UpdateStatus("대국 모드: 플레이어는 흑돌, AI는 백돌입니다.");
        }

        private void Form1_KeyDown(object? sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.R)
            {
                if (playMode == PlayMode.Training)
                {
                    UpdateStatus("학습 중에는 새 게임을 시작할 수 없습니다.");
                    return;
                }

                StartNewGame();
            }
            else if (e.KeyCode == Keys.Back)
            {
                UndoLastMove();
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            trainingCts?.Cancel();
            try
            {
                trainingTask?.Wait(2000);
            }
            catch
            {
            }

            try
            {
                saveWorkerTask?.Wait(2000);
            }
            catch
            {
            }

            base.OnFormClosing(e);
        }

        private static bool TryGetBoardPosition(Point click, out int boardX, out int boardY)
        {
            boardX = (int)Math.Round((click.X - BoardLeft) / (double)CellSize);
            boardY = (int)Math.Round((click.Y - BoardTop) / (double)CellSize);

            if (!OmokEnvironment.IsInBounds(boardX, boardY))
            {
                return false;
            }

            int centerX = BoardLeft + boardX * CellSize;
            int centerY = BoardTop + boardY * CellSize;
            int dx = click.X - centerX;
            int dy = click.Y - centerY;
            int threshold = CellSize / 2;

            return dx * dx + dy * dy <= threshold * threshold;
        }

        private static Point ToPoint((int x, int y) p)
        {
            return new Point(p.x, p.y);
        }
    }
}
