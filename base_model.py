import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

def calc_ic(pred, label):
    """计算预测值和标签之间的IC和RankIC
    
    Args:
        pred (array-like): 预测值
        label (array-like): 真实标签
        
    Returns:
        tuple: (ic, ric)
            - ic: 皮尔逊相关系数(IC)
            - ric: 斯皮尔曼相关系数(RankIC) 
    """
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    """对数据进行z-score标准化
    
    Args:
        x (torch.Tensor): 输入张量
        
    Returns:
        torch.Tensor: 标准化后的张量,满足均值为0,标准差为1
    """
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    """去除极值,保留中间95%的数据
    
    Args:
        x (torch.Tensor): 输入张量
        
    Returns:
        tuple: (mask, filtered_x)
            - mask: 布尔掩码,标记保留的数据位置
            - filtered_x: 去除极值后的数据
    """
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    """去除缺失值(NaN)
    
    Args:
        x (torch.Tensor): 输入张量
        
    Returns:
        tuple: (mask, filtered_x)
            - mask: 布尔掩码,标记非NaN的位置
            - filtered_x: 去除NaN后的数据
    """
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    """DataLoader批采样器，按日期采样数据，每个batch包含一天的数据。
    此采样器可以将数据按天进行批量采样，保证每个批次包含完整的一天数据。可以选择是否对天数顺序进行随机打乱。
    参数:
        data_source (Dataset): 数据源，需要实现get_index()方法返回带有datetime索引的数据
        shuffle (bool, optional): 是否打乱天数顺序。默认为False
    属性:
        data_source: 数据源对象
        shuffle: 是否打乱顺序的标志
        daily_count: 每天的样本数量
        daily_index: 每天数据的起始索引位置
    示例:
        >>> dataset = MyDataset()  # 自定义数据集
        >>> sampler = DailyBatchSamplerRandom(dataset, shuffle=True)
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    注意:
        - data_source必须实现get_index()方法，返回带有datetime索引的数据
        - 采样器会保持每天数据的完整性，即同一天的数据会在同一个batch中
    """

    def __init__(self, data_source, shuffle=False):
        """初始化每日批次采样器
        
        Args:
            data_source: 数据源,需要实现get_index()方法返回带datetime索引的数据
            shuffle: 是否随机打乱天数顺序,默认False
        """
        self.data_source = data_source
        self.shuffle = shuffle
        # 计算每天的样本数量
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        # 计算每天数据的起始索引位置,将累计和数组向右移动一位,第一个位置填0
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        """返回迭代器,生成每个批次的索引
        
        如果shuffle=True,会随机打乱天数顺序
        否则按时间顺序返回每天的数据索引
        
        Yields:
            np.ndarray: 当前批次数据的索引数组
        """
        if self.shuffle:
            # 随机打乱天数顺序
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            # 按顺序返回每天的数据索引
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '',use_amp=False):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler(enabled=use_amp)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            # data.shape: (N, T, F)
            # N - 股票数量
            # T - 回溯窗口长度，8
            # F - 158个因子 + 63个市场信息 + 1个标签
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label) # CSZscoreNorm

            self.train_optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(feature.float())
                    loss = self.loss_fn(pred, label)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.scaler.step(self.train_optimizer)
                self.scaler.update()
            else:
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label)
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.train_optimizer.step()
                
            losses.append(loss.item())

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        """进行模型的测试阶段。
        对给定的数据加载器中的数据进行模型评估，计算并返回平均损失值。
        Args:
            data_loader (DataLoader): 包含测试数据的数据加载器
        Returns:
            float: 测试阶段的平均损失值
        Notes:
            - 模型会被设置为评估模式 (eval mode)
            - 特征数据和标签会被移到指定设备上
            - 标签数据会进行 z-score 标准化
            - 如果启用了混合精度训练(AMP)，预测过程会在自动混合精度下进行
            - 在整个测试过程中不计算梯度
        """
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            label = zscore(label)
            
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(feature.float())
                        loss = self.loss_fn(pred, label)
                else:
                    pred = self.model(feature.float())
                    loss = self.loss_fn(pred, label)
                losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        """初始化数据加载器
        Args:
            data: 数据集
            shuffle (bool): 是否打乱数据，默认True
            drop_last (bool): 是否丢弃最后不完整的批次，默认True
        Returns:
            DataLoader: 初始化好的数据加载器
        """
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        """加载预训练的模型参数
        Args:
            param_path (str): 模型参数文件路径
        Notes:
            - 将参数加载到当前设备
            - 设置模型状态为'Previously trained'
        """
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        """
        模型训练函数
        参数:
            dl_train (DataLoader): 训练数据加载器
            dl_valid (DataLoader, optional): 验证数据加载器，默认为 None
        功能说明:
            - 使用训练数据对模型进行训练 
            - 如果提供验证数据,每个epoch会计算验证集上的IC、ICIR、RankIC、RankICIR等指标
            - 当训练损失低于阈值时会保存最佳模型参数
            - 训练过程中会打印每个epoch的训练损失和验证指标(如果有验证集)
        返回:
            None
        示例:
            >>> model.fit(train_loader, valid_loader)
            Epoch 0, train_loss 0.123456, valid ic 0.1234, icir 0.123, rankic 0.1234, rankicir 0.123.
        """
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss))
        
            if train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
                break

    def predict(self, dl_test):
        """
        根据训练好的模型对测试数据进行预测。
        参数:
            dl_test: QlibDataLoader
                测试数据加载器实例
        返回:
            tuple: (predictions, metrics)
                - predictions: pd.Series
                    预测结果,索引与输入数据对应
                - metrics: dict
                    包含以下评估指标的字典:
                    - IC: float
                        信息系数(Information Coefficient)均值
                    - ICIR: float  
                        信息系数IR比率(IC均值/IC标准差)
                    - RIC: float
                        排序信息系数(Ranked Information Coefficient)均值
                    - RICIR: float
                        排序信息系数IR比率(RIC均值/RIC标准差)
        异常:
            ValueError: 
                如果模型尚未训练完成(self.fitted<0)则抛出此异常
        注意:
            - 预测过程中会自动忽略标签中的NaN值
            - zscore标准化不会影响基于排名的评估指标的结果
        """
        if self.fitted != 'Previously trained.':
            if self.fitted < 0:
                raise ValueError("model is not fitted yet!")
            else:
                print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]
            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.
            # nan标签在计算指标时会被自动忽略
            # zscore标准化不会影响基于排名的指标的结果

            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(feature.float())
                else:
                    pred = self.model(feature.float())
                pred = pred.detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }

        return predictions, metrics
