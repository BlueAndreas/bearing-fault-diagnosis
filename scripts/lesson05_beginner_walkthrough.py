"""第五课慢速练习：默认只读训练数据；--training-demo 增加独立数字更新例子。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-pause',action='store_true',help='自动连续运行，不等回车')
    parser.add_argument('--batch-size',type=int,choices=[32,64],default=32)
    parser.add_argument('--training-demo',action='store_true',help='增加独立的旋钮参数更新例子')
    args=parser.parse_args()
    torch.set_num_threads(1)
    def stop():
        if not args.no_pause:
            try:input('\n看懂这一屏后按回车继续；可以慢慢看。')
            except EOFError:raise SystemExit('当前终端不支持输入，请使用 --no-pause。')
    def screen(title,body):
        print('\n'+'='*55+'\n'+title+'\n'+body,flush=True)
    screen('开始：先弄懂“题目和答案”，再认识 PyTorch 名字',
           '今天默认只观察数据，不训练 CNN。\n输入 X 是振动数字；答案 y 是已知类别编号。\n0=正常，1=内圈，2=外圈，3=滚动体。\n小例子每段只用 4 点；真实项目每段是 1024 点。')
    stop()
    values=torch.tensor([[.1,.2,-.3,.4],[.5,-.2,.3,-.1],[-.2,.1,.6,-.4]],dtype=torch.float32)
    labels=torch.tensor([0,1,2],dtype=torch.long)
    screen('第 1 步：3 张“题目＋答案”卡',
           '\n'.join(f'卡 {i}：振动 {[round(float(v),2) for v in values[i]]} → 类别 {int(labels[i])}' for i in range(3))+
           '\n这些演示数字是人为编写的，不是实际故障证据。\nTensor 就是 PyTorch 存放这些数字的数组。\n振动带小数，用 float32；类别编号用 long 整数。')
    stop()
    signals=values.unsqueeze(1)
    assert torch.equal(signals[:,0,:],values)
    screen('第 2 步：形状是“有几份、每份怎样排”',
           f'原来形状 {tuple(values.shape)}：3 个样本，每个 4 点。\n加通道后 {tuple(signals.shape)}：3 个样本，每个 1 个通道，每通道 4 点。\n第 0 个样本形状 {tuple(signals[0].shape)}：1 个通道、4 点。\nunsqueeze(1) 只加一个长度为 1 的维度；振动值完全相同。')
    stop()
    dataset=TensorDataset(signals,labels)
    x1,y1=dataset[1]
    assert torch.equal(x1,signals[1]) and int(y1)==1
    screen('第 3 步：Dataset 按编号拿出配对的数据',
           f'len(dataset) = {len(dataset)}：总共 3 张卡。\ndataset[1] 返回卡 1：\n振动 {[[round(float(v),2) for v in x1[0]]]}\n答案 {int(y1)}（内圈）。\nPython 编号从 0 开始，所以编号 1 是第二张。\nDataset 不会训练模型，只管按编号取样本和标签。')
    stop()
    batches=list(DataLoader(dataset,batch_size=2,shuffle=False,drop_last=False,num_workers=0))
    assert [len(y) for x,y in batches]==[2,1]
    screen('第 4 步：DataLoader 每次打包一批卡',
           '\n'.join(f'第 {i+1} 批：X 形状 {tuple(x.shape)}，标签 {y.tolist()}' for i,(x,y) in enumerate(batches))+
           '\n第一批是卡 0、卡 1；最后剩卡 2，也保留。\n这里不打乱，是为了让你看清编号和答案。\n真实训练可以打乱卡片顺序，但卡片上的采样点仍按时间排列。')
    stop()
    path=ROOT/'data/processed/lesson03/dataset.npz'
    if not path.is_file():raise SystemExit('缺少第三课 dataset.npz，请先准备第三课数据。')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as z:
        raw=z['X_train'].copy(); targets=z['y_train'].copy()
    assert raw.shape==(368,1024) and targets.shape==(368,)
    x=torch.from_numpy(raw.astype(np.float32,copy=False)).unsqueeze(1)
    y=torch.from_numpy(targets.astype(np.int64,copy=False))
    real=TensorDataset(x,y)
    loader=DataLoader(real,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=0)
    actual=list(loader)
    sizes=[len(by) for bx,by in actual]
    assert sum(sizes)==368
    assert torch.equal(torch.cat([bx for bx,by in actual]),x)
    assert torch.equal(torch.cat([by for bx,by in actual]),y)
    screen('第 5 步：换成项目里的真实训练数据',
           f'整体 X = {tuple(x.shape)}，整体 y = {tuple(y.shape)}。\n单个窗口 X = {tuple(real[0][0].shape)}，单个答案 = {int(real[0][1])}。\n第一批 X = {tuple(actual[0][0].shape)}，y = {tuple(actual[0][1].shape)}。\n每批数量：{sizes}\n共 {len(loader)} 批，最后 {sizes[-1]} 个，总样本仍是 368 个。\n本练习为便于核对使用 shuffle=False；前面的原训练演示用 True。\nX_train 已按训练集共享参数标准化，这里不重新拟合。')
    summary={'batch_size':args.batch_size,'toy_batch_sizes':[len(y) for x,y in batches],
             'real_input_shape':list(x.shape),'first_batch_shape':list(actual[0][0].shape),
             'batch_sizes':sizes,'samples':sum(sizes),'sample_and_label_order_verified':True,
             'input_values_unchanged_after_unsqueeze':True,'shuffle':False,
             'cnn_training_performed':False,'torch_version':torch.__version__}
    stop()
    if args.training_demo:
        screen('第二轮：只调整一个数字，不训练轴承分类模型',
               '把参数 w 当作旋钮。初始 w=2，目标是 5。\n误差用 loss=(w-5)²，初始误差为 9。\n这是平方误差小例子；实际四分类模型使用交叉熵。')
        stop()
        w=torch.tensor(2.,requires_grad=True)
        optimizer=torch.optim.SGD([w],lr=.1)
        optimizer.zero_grad()
        loss=(w-5)**2
        loss.backward()
        assert float(w.detach())==2. and float(w.grad)==-6.
        screen('第二轮第 1 步：backward 只算梯度',
               f'loss = {float(loss.detach()):.2f}\n梯度 w.grad = {float(w.grad):.2f}\n但 w 仍然是 {float(w.detach()):.2f}。\n先不推导 -6 怎样算出，只观察：参数还没变。\n在这个位置，增加 w 能减小误差；SGD 会减去负梯度。')
        stop()
        optimizer.step()
        after=float(w.detach()); after_loss=float(((w-5)**2).detach())
        assert abs(after-2.6)<1e-6 and abs(after_loss-5.76)<1e-5
        screen('第二轮第 2 步：step 才真正更新参数',
               f'w = 2 - 0.1 × (-6) = {after:.2f}\n更新后误差 = {after_loss:.2f}\n从 2 向目标 5 靠近了一点。\n记住：backward 算怎么调，step 执行调整。\n普通的一批一更新流程，先 zero_grad 清理旧梯度。')
        summary['scalar_demo']={'before':2.,'target':5.,'gradient':-6.,'learning_rate':.1,
                                'after':after,'loss_before':9.,'loss_after':after_loss}
        stop()
    after_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    assert before==after_hash
    summary['dataset_sha256']=before
    summary['dataset_file_unchanged']=True
    out=ROOT/'outputs/lesson05/beginner'
    out.mkdir(parents=True,exist_ok=True)
    name=f'walkthrough-batch-{args.batch_size}'+('-with-update' if args.training_demo else '')+'.json'
    (out/name).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    screen('完成：先用自己的话回答这 3 个问题',
           f'1. Dataset 和 DataLoader 各负责什么？\n2. ({args.batch_size}, 1, 1024) 的三个数字是什么意思？\n3. 每批最多 {args.batch_size} 个，368 个样本为什么最后一批只有 {sizes[-1]} 个？\n本次只新增慢速练习记录；没有改动原数据或已保存 CNN/SVM。\n结果：'+str(out/name))

if __name__=='__main__':main()
