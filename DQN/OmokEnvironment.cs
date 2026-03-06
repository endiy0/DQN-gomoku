namespace DQN
{
    public enum StoneColor
    {
        None = 0,
        Black = 1,
        White = 2
    }

    public readonly record struct OmokStepResult(
        bool IsValid,
        bool IsGameOver,
        StoneColor Winner,
        float Reward,
        int ActionIndex,
        int X,
        int Y,
        string Message
    );

    public sealed class OmokEnvironment
    {
        public const int BoardSize = 15;
        public const int ActionCount = BoardSize * BoardSize;

        private readonly StoneColor[,] board = new StoneColor[BoardSize, BoardSize];
        private readonly List<int> moveHistory = [];

        public float InvalidMoveReward { get; set; } = -1f;
        public float WinReward { get; set; } = 1f;
        public float DrawReward { get; set; } = 0f;
        public float MoveReward { get; set; } = 0f;

        public StoneColor CurrentTurn { get; private set; } = StoneColor.Black;
        public StoneColor Winner { get; private set; } = StoneColor.None;
        public bool IsGameOver { get; private set; }
        public int? LastActionIndex { get; private set; }
        public int MoveCount => moveHistory.Count;

        public void Reset()
        {
            Array.Clear(board, 0, board.Length);
            moveHistory.Clear();
            CurrentTurn = StoneColor.Black;
            Winner = StoneColor.None;
            IsGameOver = false;
            LastActionIndex = null;
        }

        public StoneColor[,] GetBoardCopy()
        {
            var copied = new StoneColor[BoardSize, BoardSize];
            Array.Copy(board, copied, board.Length);
            return copied;
        }

        public StoneColor GetStoneAt(int x, int y)
        {
            EnsureInBounds(x, y);
            return board[x, y];
        }

        public bool CanPlaceAt(int x, int y)
        {
            return IsInBounds(x, y) && !IsGameOver && board[x, y] == StoneColor.None;
        }

        public bool CanPlaceAction(int actionIndex)
        {
            if (!TryToBoardPosition(actionIndex, out int x, out int y))
            {
                return false;
            }

            return CanPlaceAt(x, y);
        }

        public OmokStepResult StepByAction(int actionIndex)
        {
            TryStepByAction(actionIndex, out OmokStepResult result);
            return result;
        }

        public bool TryStepByAction(int actionIndex, out OmokStepResult result)
        {
            if (!TryToBoardPosition(actionIndex, out int x, out int y))
            {
                result = BuildInvalidResult(actionIndex, -1, -1, "유효하지 않은 액션 인덱스입니다.");
                return false;
            }

            return TryStepAt(x, y, out result);
        }

        public OmokStepResult StepAt(int x, int y)
        {
            TryStepAt(x, y, out OmokStepResult result);
            return result;
        }

        public bool TryStepAt(int x, int y, out OmokStepResult result)
        {
            int actionIndex = IsInBounds(x, y) ? ToActionIndex(x, y) : -1;

            if (!IsInBounds(x, y))
            {
                result = BuildInvalidResult(actionIndex, x, y, "보드 범위를 벗어난 좌표입니다.");
                return false;
            }

            if (IsGameOver)
            {
                result = BuildInvalidResult(actionIndex, x, y, "이미 종료된 게임입니다.");
                return false;
            }

            if (board[x, y] != StoneColor.None)
            {
                result = BuildInvalidResult(actionIndex, x, y, "이미 돌이 놓인 자리입니다.");
                return false;
            }

            board[x, y] = CurrentTurn;
            moveHistory.Add(actionIndex);
            LastActionIndex = actionIndex;

            if (IsWinningMove(x, y, CurrentTurn))
            {
                IsGameOver = true;
                Winner = CurrentTurn;
                result = new OmokStepResult(
                    IsValid: true,
                    IsGameOver: true,
                    Winner: Winner,
                    Reward: WinReward,
                    ActionIndex: actionIndex,
                    X: x,
                    Y: y,
                    Message: $"{GetStoneName(Winner)} 승리");
                return true;
            }

            if (moveHistory.Count >= ActionCount)
            {
                IsGameOver = true;
                Winner = StoneColor.None;
                result = new OmokStepResult(
                    IsValid: true,
                    IsGameOver: true,
                    Winner: StoneColor.None,
                    Reward: DrawReward,
                    ActionIndex: actionIndex,
                    X: x,
                    Y: y,
                    Message: "무승부");
                return true;
            }

            CurrentTurn = GetOppositeStone(CurrentTurn);
            result = new OmokStepResult(
                IsValid: true,
                IsGameOver: false,
                Winner: StoneColor.None,
                Reward: MoveReward,
                ActionIndex: actionIndex,
                X: x,
                Y: y,
                Message: "착수 성공");
            return true;
        }

        public bool UndoLastAction()
        {
            if (moveHistory.Count == 0)
            {
                return false;
            }

            int actionIndex = moveHistory[^1];
            moveHistory.RemoveAt(moveHistory.Count - 1);

            (int x, int y) = ToBoardPosition(actionIndex);
            StoneColor removed = board[x, y];
            board[x, y] = StoneColor.None;

            CurrentTurn = removed == StoneColor.None ? StoneColor.Black : removed;
            Winner = StoneColor.None;
            IsGameOver = false;
            LastActionIndex = moveHistory.Count > 0 ? moveHistory[^1] : null;

            return true;
        }

        public List<int> GetLegalActions()
        {
            var legal = new List<int>(ActionCount - moveHistory.Count);

            if (IsGameOver)
            {
                return legal;
            }

            for (int y = 0; y < BoardSize; y++)
            {
                for (int x = 0; x < BoardSize; x++)
                {
                    if (board[x, y] == StoneColor.None)
                    {
                        legal.Add(ToActionIndex(x, y));
                    }
                }
            }

            return legal;
        }

        public bool[] GetLegalActionMask()
        {
            var mask = new bool[ActionCount];

            if (IsGameOver)
            {
                return mask;
            }

            for (int actionIndex = 0; actionIndex < ActionCount; actionIndex++)
            {
                (int x, int y) = ToBoardPosition(actionIndex);
                mask[actionIndex] = board[x, y] == StoneColor.None;
            }

            return mask;
        }

        public float[] GetObservationVector()
        {
            return GetObservationVector(CurrentTurn);
        }

        public float[] GetObservationVector(StoneColor perspective)
        {
            if (perspective == StoneColor.None)
            {
                perspective = CurrentTurn;
            }

            StoneColor opponent = GetOppositeStone(perspective);
            var observation = new float[ActionCount];

            for (int y = 0; y < BoardSize; y++)
            {
                for (int x = 0; x < BoardSize; x++)
                {
                    int actionIndex = ToActionIndex(x, y);
                    StoneColor cell = board[x, y];

                    observation[actionIndex] = cell switch
                    {
                        StoneColor.None => 0f,
                        _ when cell == perspective => 1f,
                        _ when cell == opponent => -1f,
                        _ => 0f
                    };
                }
            }

            return observation;
        }

        public float[] GetTurnAwareObservation()
        {
            var observation = new float[ActionCount + 1];
            float[] boardObservation = GetObservationVector();
            Array.Copy(boardObservation, observation, boardObservation.Length);
            observation[^1] = CurrentTurn == StoneColor.Black ? 1f : -1f;
            return observation;
        }

        public string GetStatusText()
        {
            if (IsGameOver)
            {
                return Winner == StoneColor.None ? "무승부" : $"{GetStoneName(Winner)} 승리";
            }

            return $"현재 차례: {GetStoneName(CurrentTurn)}";
        }

        public static bool IsInBounds(int x, int y)
        {
            return x >= 0 && x < BoardSize && y >= 0 && y < BoardSize;
        }

        public static int ToActionIndex(int x, int y)
        {
            if (!IsInBounds(x, y))
            {
                throw new ArgumentOutOfRangeException(nameof(x), "보드 범위를 벗어난 좌표입니다.");
            }

            return y * BoardSize + x;
        }

        public static (int x, int y) ToBoardPosition(int actionIndex)
        {
            if (!TryToBoardPosition(actionIndex, out int x, out int y))
            {
                throw new ArgumentOutOfRangeException(nameof(actionIndex), "유효하지 않은 액션 인덱스입니다.");
            }

            return (x, y);
        }

        public static bool TryToBoardPosition(int actionIndex, out int x, out int y)
        {
            x = -1;
            y = -1;

            if (actionIndex < 0 || actionIndex >= ActionCount)
            {
                return false;
            }

            x = actionIndex % BoardSize;
            y = actionIndex / BoardSize;
            return true;
        }

        public static StoneColor GetOppositeStone(StoneColor stone)
        {
            return stone == StoneColor.Black ? StoneColor.White : StoneColor.Black;
        }

        public static string GetStoneName(StoneColor stone)
        {
            return stone switch
            {
                StoneColor.Black => "흑",
                StoneColor.White => "백",
                _ => "없음"
            };
        }

        private bool IsWinningMove(int x, int y, StoneColor stone)
        {
            return CountConnected(x, y, 1, 0, stone) >= 5
                || CountConnected(x, y, 0, 1, stone) >= 5
                || CountConnected(x, y, 1, 1, stone) >= 5
                || CountConnected(x, y, 1, -1, stone) >= 5;
        }

        private int CountConnected(int x, int y, int dx, int dy, StoneColor stone)
        {
            return 1
                + CountDirection(x, y, dx, dy, stone)
                + CountDirection(x, y, -dx, -dy, stone);
        }

        private int CountDirection(int x, int y, int dx, int dy, StoneColor stone)
        {
            int count = 0;
            int nx = x + dx;
            int ny = y + dy;

            while (IsInBounds(nx, ny) && board[nx, ny] == stone)
            {
                count++;
                nx += dx;
                ny += dy;
            }

            return count;
        }

        private OmokStepResult BuildInvalidResult(int actionIndex, int x, int y, string message)
        {
            return new OmokStepResult(
                IsValid: false,
                IsGameOver: IsGameOver,
                Winner: Winner,
                Reward: InvalidMoveReward,
                ActionIndex: actionIndex,
                X: x,
                Y: y,
                Message: message);
        }

        private static void EnsureInBounds(int x, int y)
        {
            if (!IsInBounds(x, y))
            {
                throw new ArgumentOutOfRangeException(nameof(x), "보드 범위를 벗어난 좌표입니다.");
            }
        }
    }
}
