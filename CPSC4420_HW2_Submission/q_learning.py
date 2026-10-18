"""CPSC 4420 HW2: tabular Q-learning on CliffWalking-v1.
Install: pip install -r requirements.txt; run: python q_learning.py
The Gymnasium environment is used when installed. The local fallback reproduces
4x12 deterministic cliff-walking rules for offline reproducibility.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
ACTIONS = [( -1,0),(0,1),(1,0),(0,-1)] # up/right/down/left

class LocalCliffWalking:
    observation_space_n, action_space_n = 48, 4
    def __init__(self): self.s = 36
    def reset(self, seed=None): self.s=36; return self.s, {}
    def step(self, action):
        r,c=divmod(self.s,12)
        dr,dc=ACTIONS[action]
        r=min(3,max(0,r+dr)); c=min(11,max(0,c+dc))
        state=r*12+c
        if r==3 and 1<=c<=10: state=36; reward=-100
        else: reward=-1
        self.s=state
        terminated=state==47
        return state,reward,terminated,False,{}
    def close(self): pass

def make_env():
    try:
        import gymnasium as gym
        return gym.make('CliffWalking-v1'), 'Gymnasium CliffWalking-v1'
    except ImportError:
        return LocalCliffWalking(), 'Local faithful CliffWalking rules (Gymnasium not installed)'

def greedy_action(q, state):
    # deterministic tie breaking, helps reproducibility
    return int(np.argmax(q[state]))

def fit(alpha=.5, gamma=.99, eps_decay=.997, episodes=3000, seed=42,
        eps_start=1.0, eps_min=.05, action_noise=0.0):
    env, source=make_env(); rng=np.random.default_rng(seed)
    q=np.zeros((48,4),dtype=float); rewards=[]; steps=[]; eps=eps_start
    for episode in range(episodes):
        s,_=env.reset(seed=seed+episode); total=0; count=0
        for t in range(200):
            a=int(rng.integers(4)) if rng.random()<eps else greedy_action(q,s)
            # On action corruption choose one of the THREE OTHER valid actions.
            if rng.random()<action_noise: a=(a+int(rng.integers(1,4)))%4
            sn,r,done,truncated,_=env.step(a)
            target=r if done or truncated else r+gamma*np.max(q[sn])
            q[s,a]+=alpha*(target-q[s,a])
            s=sn; total+=r; count+=1
            if done or truncated: break
        rewards.append(total); steps.append(count)
        eps=max(eps_min,eps*eps_decay)
    env.close()
    return q,np.asarray(rewards),np.asarray(steps),source

def evaluate(q, episodes=100, seed=1234, action_noise=0):
    env,_=make_env(); rng=np.random.default_rng(seed); rewards=[]; steps=[]; wins=0
    for i in range(episodes):
        s,_=env.reset(seed=seed+i); ret=0
        for t in range(200):
            a=greedy_action(q,s)
            if rng.random()<action_noise:a=(a+int(rng.integers(1,4)))%4
            s,r,done,truncated,_=env.step(a)
            ret+=r
            if done or truncated: break
        rewards.append(ret);steps.append(t+1);wins+=bool(done)
    env.close()
    return {'mean_reward':float(np.mean(rewards)), 'std_reward':float(np.std(rewards)),
            'success_rate':wins/episodes,'mean_steps':float(np.mean(steps))}

def smooth(y,window=100):
    return np.convolve(y,np.ones(window)/window,mode='valid')

def plot_curves(curves,title,name):
    plt.figure(figsize=(9.5,4.6))
    for label,data in curves:
        z=smooth(data); plt.plot(np.arange(len(z))+100,z,label=label,linewidth=1.6)
    plt.xlabel('Training episode');plt.ylabel('Mean episodic reward (100-episode moving average)')
    plt.title(title);plt.legend();plt.grid(alpha=.22);plt.tight_layout()
    plt.savefig(OUT/name,dpi=170);plt.close()

def draw_frame(s, label):
    from PIL import Image,ImageDraw,ImageFont
    im=Image.new('RGB',(720,330),'#f5f7fa');d=ImageDraw.Draw(im)
    cell=48;ox=72;oy=53
    for row in range(4):
        for col in range(12):
            xy=(ox+col*cell,oy+row*cell,ox+(col+1)*cell-2,oy+(row+1)*cell-2)
            color='#f1d3d3' if row==3 and 1<=col<=10 else '#dcecf4'
            if (row,col)==(3,0):color='#bbd7af'
            if (row,col)==(3,11):color='#f5dc92'
            d.rectangle(xy,fill=color)
    for col in range(1,11):d.text((ox+col*cell+17,oy+3*cell+16),'×',fill='#993c3c')
    d.text((ox+8,oy+3*cell+16),'S',fill='#205b2f')
    d.text((ox+11*cell+15,oy+3*cell+16),'G',fill='#745512')
    r,c=divmod(s,12);cx=ox+c*cell+23;cy=oy+r*cell+23
    d.ellipse((cx-15,cy-15,cx+15,cy+15),fill='#2558a7',outline='white',width=3)
    d.text((72,270),label,fill='#152b48')
    return np.array(im)

def make_video(q,name='trained_agent.mp4'):
    env,_=make_env();s,_=env.reset(); frames=[]; states=[s]
    for t in range(200):
        a=greedy_action(q,s)
        s,r,done,truncated,_=env.step(a);states.append(s)
        if done or truncated:break
    # 2 fps with hold on final frame
    for i,st in enumerate(states):frames.extend([draw_frame(st,f'Greedy trained policy · step {i}')]*3)
    frames.extend([draw_frame(states[-1],f'Finished after {len(states)-1} steps')]*12)
    with imageio.get_writer(OUT/name,fps=6,codec='libx264',quality=7,macro_block_size=16) as writer:
        for frame in frames:writer.append_data(frame)
    env.close();return len(states)-1

def main():
    configs={'baseline':dict(alpha=.5,gamma=.99,eps_decay=.997)}
    for a in [.1,.5,.9]: configs[f'alpha_{a}']=dict(alpha=a,gamma=.99,eps_decay=.997)
    for g in [.5,.9,.99]:configs[f'gamma_{g}']=dict(alpha=.5,gamma=g,eps_decay=.997)
    for e in [.99,.997,.999]:configs[f'decay_{e}']=dict(alpha=.5,gamma=.99,eps_decay=e)
    records=[];stored={};curves={}
    for key,params in configs.items():
        q,rew,steps,source=fit(**params)
        metrics=evaluate(q)
        stored[key]=q;curves[key]=rew
        np.save(OUT/f'{key}_q.npy',q)
        records.append({'configuration':key,**params,**metrics,'train_last100_mean':float(np.mean(rew[-100:]))})
        print(key,'last100:',round(np.mean(rew[-100:]),2),'evaluation:',metrics,flush=True)
    # best by independently evaluated mean reward, then success rate, then steps.
    best=max(records,key=lambda x:(x['mean_reward'],x['success_rate'],-x['mean_steps']))
    bestkey=best['configuration'];qbest=stored[bestkey]
    plot_curves([('Baseline',curves['baseline'])],'Baseline learning curve','baseline.png')
    for prefix,vals,ttl,fn in [('alpha',[.1,.5,.9],'Learning-rate comparison','learning_rate.png'),
                                ('gamma',[.5,.9,.99],'Discount-factor comparison','discount_factor.png'),
                                ('decay',[.99,.997,.999],'Exploration-decay comparison','epsilon_decay.png')]:
        plot_curves([(f'{prefix}={val}',curves[f'{prefix}_{val}']) for val in vals],ttl,fn)
    noise_rows=[];noise_curves=[]
    for p in [0,.10,.25]:
        q,r,_,_=fit(action_noise=p)
        own=evaluate(q,action_noise=p)
        transferred=evaluate(qbest,action_noise=p)
        noise_rows.append({'noise':p,**own,**{'original_'+k:v for k,v in transferred.items()},'train_last100_mean':float(np.mean(r[-100:]))})
        noise_curves.append((f'p={p}',r))
    plot_curves(noise_curves,'Action-noise comparison','noise.png')
    with open(OUT/'metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    with open(OUT/'noise_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=noise_rows[0].keys());w.writeheader();w.writerows(noise_rows)
    steps=make_video(qbest)
    (OUT/'run_info.txt').write_text(f'Environment: {source}\nSelected by highest evaluation mean reward: {bestkey}\nVideo steps: {steps}\n')
    print('BEST',bestkey,'SOURCE',source,'VIDEO STEPS',steps)
if __name__=='__main__':main()
