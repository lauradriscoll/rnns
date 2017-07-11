%% init
tau = 10; %ms
n_all = 100;
n_its = 1000;
n_in = 3;
pc = .1;
I0 = .001;
n_train_its = 60;
n_out = 1;
x = randn(n_all,1);
r = tanh(x);
dx = randn(n_all,1);
ei = nan(n_train_its,n_in,n_all,n_its);
eo = nan(n_train_its,n_in,n_out,n_its);
Wr = randn(n_all)/sqrt(pc*n_all);
Wi = randn(n_all,n_in);
Wo = randn(n_all,n_out)/sqrt(n_all);
% Wfb = rand(N,1)*2-1;
r_train = cell(n_in,1);
r_test = cell(n_in,1);
r_init = cell(n_in,1);
R_all = cell(n_in,1);
n_plastic = round(.6*n_all);
i_plastic = sort(randperm(n_all,n_plastic));
Po_cell = cell(n_out,n_in);
for n = 1:n_out
    for ni = 1:n_in
    Po_cell{n,ni} = eye(n_all)/(n_all);
    end
end

stim_dur = 50;
out_dur = 50;
train_window = 200;
t_on = 200;
t_off = t_on+stim_dur;
t_end = t_off+train_window;
y = zeros(n_its,n_in);
y(t_on:t_off,1) = 5;
y(t_on:t_off,2) = -5;

f = ones(n_its,2)*-.5;
f(:,2) = f(:,2)+1;
f([1:t_on t_end:end],:) = 0;

% tc = gauspuls('cutoff',100,.5,[],-40); 
% t = -tc : tc/(t_off-t_on) : tc; 
% yi = gauspuls(t,100,0.5); 
% f((t_end-size(yi,2)+1):t_end,1) = .5*(-yi)-.5;
% f((t_end-size(yi,2)+1):t_end,2) = .5*(yi)+.5;

%% record innate target
for in_idx_train = 1:2
R_all{in_idx_train} = nan(n_all,n_its);
for t = 1:n_its
    for i = 1:n_all
        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,in_idx_train)*y(t,in_idx_train)');
    end
    x = x+dx;
    r = tanh(x);
    z = Wo'*r;
    R_all{in_idx_train}(:,t) = r;
end
end

%% pre train output
for in_idx_train = 1:2
I_noise = randn(n_all,n_its)*I0;    
r_init{in_idx_train} = nan(n_all,n_its);
for t = 1:n_its
    for i = 1:n_all
        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,in_idx_train)*y(t,in_idx_train)' + I_noise(i,t));
    end
    x = x+dx;
    r = tanh(x);
    z = Wo'*r;
    r_init{in_idx_train}(:,t) = r;
end
end

%% training
for in_idx_train = 1:2
figure('position',[1000 100 800 900]);
subplot(3,2,1)
imagesc(R_all{1})
xlabel('Time (ms)')
ylabel('Cells')
title('Trial Type 1')
subplot(3,2,2)
imagesc(R_all{2})
xlabel('Time (ms)')
ylabel('Cells')
title('Trial Type 2')
size_P = nan(n_train_its,t_end,n_all);

for l = 1:n_train_its
I_noise = randn(n_all,n_its)*I0;    
r_train{in_idx_train} = nan(n_all,n_its);
x = randn(n_all,1);
r = tanh(x);
Wr = Wr + randn(n_all)*I0;

for t = t_on:t_end
    r_train{in_idx_train}(:,t) = r;
    
    %innate training
    ei(l,in_idx_train,:,t) = r_train{in_idx_train}(:,t) - R_all{in_idx_train}(:,t);    
    
    for i = 1:n_all        
        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,in_idx_train)*y(t,in_idx_train)' + I_noise(i,t)); %+ Wfb(i)*z
    end
    x = x+dx;
    r = tanh(x);   
    
    eo(l,in_idx_train,:,t) = Wo'*r - f(t,in_idx_train);    
    for n = 1:n_out
        Po = Po_cell{n,in_idx_train};
        Po_cell{n,in_idx_train} = Po - (Po*r*r'*Po)/(1+r'*Po*r);
        Wo(:,n) = Wo(:,n) - eo(l,in_idx_train,n,t) * Po_cell{n,in_idx_train}*r;
    end
end
end


%% post train output
for in_idx = 1:2
I_noise = randn(n_all,n_its)*I0;    
r_test{in_idx} = nan(n_all,n_its);
for t = 1:n_its
    for i = 1:n_all
        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,in_idx)*y(t,in_idx)' + I_noise(i,t));
    end
    x = x+dx;
    r = tanh(x);
    z = Wo'*r;
    r_test{in_idx}(:,t) = r;
end
end

for in_idx = 1:2
for sp = 1:3
subplot(7,2,6+2*sp+in_idx-2)
hold on
h1 = area([t_on t_end],[1 1]);
h2 = area([t_on t_end],[-1 -1]);
h1(:).FaceColor = [.7 .7 .7];
h1(:).EdgeColor = 'none';
h1(:).FaceAlpha = .5;
h2(:).FaceColor = [.7 .7 .7];
h2(:).EdgeColor = 'none';
h2(:).FaceAlpha = .5;
plot(R_all{in_idx}(sp,:),'b')
plot(r_test{in_idx}(sp,:),'r')
plot([t_on t_on],[-1 1],'-k')
plot([t_off t_off],[-1 1],'-k')
title(['Cell ' num2str(sp)])
set(gca,'xtick',[],'ytick',[])
ylabel('Event Rate')
end

subplot(7,2,12+in_idx)
hold on
h1 = area([t_on t_end],[1 1]);
h2 = area([t_on t_end],[-1 -1]);
h1(:).FaceColor = [.7 .7 .7];
h1(:).EdgeColor = 'none';
h1(:).FaceAlpha = .5;
h2(:).FaceColor = [.7 .7 .7];
h2(:).EdgeColor = 'none';
h2(:).FaceAlpha = .5;
plot(f(:,in_idx),'b')
plot(Wo'*r_test{in_idx},'r')
plot([t_on t_on],[-1 1],'-k')
plot([t_off t_off],[-1 1],'-k')
ylim([-1 1])
title('Output')
set(gca,'xtick',[],'ytick',[])
ylabel('Event Rate')
end
end