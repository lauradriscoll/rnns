%% init
tau = 10; %ms
N = 100;
n_its = 1000;
n_in = 1;
pc = .1;
I0 = .001;
n_train_its = 10;
n_out = N;
x = randn(N,1);
r = tanh(x);
dx = randn(N,1);
e = nan(n_train_its,N,n_its);
Wr = randn(N)/sqrt(pc*N);
Wi = randn(N,n_in);
Wo = randn(N,n_out)/sqrt(N);
% Wfb = rand(N,1)*2-1;
I_noise = randn(N,n_its)*I0;
r_train = nan(N,n_its);
r_test = nan(N,n_its);
r_init = nan(N,n_its);
R_all = nan(N,n_its);
n_plastic = round(.6*N);
i_plastic = sort(randperm(N,n_plastic));
P_cell = cell(N,1);
for n = 1:N
    P_cell{n} = eye(N)/(N);
end

stim_dur = 50;
train_window = 200;
t_on = 100;
t_off = 100+stim_dur;
t_end = t_off+train_window;
y = zeros(n_its,n_in);
y(t_on:t_off,1) = 5;

%% record innate target
for t = 1:n_its
    for i = 1:N
        dx(i) = 1/tau*(-x(i) + Wr(i,:)*r + Wi(i,:)*y(t,:)');
    end
    x = x+dx;
    r = tanh(x);
    z = Wo*r;
    R_all(:,t) = r;
end

%% pre train output
for t = 1:n_its
    for i = 1:N
        dx(i) = 1/tau*(-x(i) + Wr(i,:)*r + Wi(i,:)*y(t,:)' + I_noise(i));
    end
    x = x+dx;
    r = tanh(x);
    z = Wo*r;
    r_init(:,t) = r;
end

figure('position',[1000 100 800 900]);
subplot(3,1,1)
imagesc(R_all)
for sp = 1:3
subplot(9,2,6+2*sp-1)
hold on
h1 = area([t_on t_end],[1 1]);
h2 = area([t_on t_end],[-1 -1]);
h1(:).FaceColor = [.7 .7 .7];
h1(:).EdgeColor = 'none';
h1(:).FaceAlpha = .5;
h2(:).FaceColor = [.7 .7 .7];
h2(:).EdgeColor = 'none';
h2(:).FaceAlpha = .5;
plot(R_all(sp,:),'b')
plot(r_init(sp,:),'r')
plot([t_on t_on],[-1 1],'-k')
plot([t_off t_off],[-1 1],':k')
end

%% training
for l = 1:n_train_its
x = randn(N,1);
r = tanh(x);
for t = 1:t_end
    r_train(:,t) = r;
    
    %innate training
    e(l,:,t) = r_train(:,t) - R_all(:,t);    
    
    for i = 1:N           
            P = P_cell{i_plastic==i};
            P_cell{i_plastic==i} = P - (P*r*r'*P)/(1+r'*P*r);
            Wr(i,:) = Wr(i,:) - e(l,i,t) * P_cell{i_plastic==i}(i,:)*r;
            
        dx(i) = 1/tau*(-x(i) + Wr(i,:)*r + Wi(i,:)*y(t,:)' + I_noise(i)); %+ Wfb(i)*z
    end
    x = x+dx;
    r = tanh(x);
    z = Wo*r;
end
end

for sp = 1:3
subplot(9,2,6+2*sp)
hold on
h1 = area([t_on t_end],[1 1]);
h2 = area([t_on t_end],[-1 -1]);
h1(:).FaceColor = [.7 .7 .7];
h1(:).EdgeColor = 'none';
h1(:).FaceAlpha = .5;
h2(:).FaceColor = [.7 .7 .7];
h2(:).EdgeColor = 'none';
h2(:).FaceAlpha = .5;
plot(R_all(sp,:),'b')
plot(r_train(sp,:),'r')
plot([t_on t_on],[-1 1],'-k')
plot([t_off t_off],[-1 1],':k')
end

%% testing
for t = 1:n_its
    for i = 1:N
        dx(i) = 1/tau*(-x(i) + Wr(i,:)*r + Wi(i,:)*y(t,:)' + I_noise(i));
    end
    x = x+dx;
    r = tanh(x);
    z = Wo*r;
    r_test(:,t) = r;
end

for sp = 1:3
subplot(9,2,6+2*sp-1)
hold on
h1 = area([t_on t_end],[1 1]);
h2 = area([t_on t_end],[-1 -1]);
h1(:).FaceColor = [.7 .7 .7];
h1(:).EdgeColor = 'none';
h1(:).FaceAlpha = .5;
h2(:).FaceColor = [.7 .7 .7];
h2(:).EdgeColor = 'none';
h2(:).FaceAlpha = .5;
plot(R_all(sp,:),'b')
plot(r_test(sp,:),'r')
plot([t_on t_on],[-1 1],'-k')
plot([t_off t_off],[-1 1],':k')
end