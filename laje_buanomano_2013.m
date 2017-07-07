%% init
tau = 10; %ms
n_all = 100;
n_its = 1000;
n_in = 1;
pc = .1;
I0 = .001;
n_train_its = 30;
n_out = 1;
x = randn(n_all,1);
r = tanh(x);
dx = randn(n_all,1);
ei = nan(n_train_its,n_all,n_its);
eo = nan(n_train_its,n_out,n_its);
Wr = randn(n_all)/sqrt(pc*n_all);
Wi = randn(n_all,n_in);
Wo = randn(n_all,n_out)/sqrt(n_all);
% Wfb = rand(N,1)*2-1;
I_noise = randn(n_all,n_its)*I0;
r_train = nan(n_all,n_its);
r_test = nan(n_all,n_its);
r_init = nan(n_all,n_its);
R_all = nan(n_all,n_its);
n_plastic = round(.6*n_all);
i_plastic = sort(randperm(n_all,n_plastic));
Pi_cell = cell(n_all,1);
Po_cell = cell(n_out,1);
for n = 1:n_all
    Pi_cell{n} = eye(n_all)/(n_all);
end
for n = 1:n_out
    Po_cell{n} = eye(n_all)/(n_all);
end

stim_dur = 50;
out_dur = 50;
train_window = 200;
t_on = 200;
t_off = t_on+stim_dur;
t_end = t_off+train_window;
y = zeros(n_its,n_in);
y(t_on:t_off,1) = 5;
f = ones(n_its,n_in)*-.5;
f(t_end-2*out_dur:t_end-out_dur,1) = 1;

%% record innate target
for t = 1:n_its
    for i = 1:n_all
        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,:)*y(t,:)');
    end
    x = x+dx;
    r = tanh(x);
    z = Wo'*r;
    R_all(:,t) = r;
end

%% pre train output
for t = 1:n_its
    for i = 1:n_all
        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,:)*y(t,:)' + I_noise(i));
    end
    x = x+dx;
    r = tanh(x);
    z = Wo'*r;
    r_init(:,t) = r;
end

figure('position',[1000 100 800 900]);
subplot(3,1,1)
imagesc(R_all)
xlabel('Time (ms)')
ylabel('Cells')
for sp = 1:3
subplot(7,2,6+2*sp-1)
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
plot([t_off t_off],[-1 1],'-k')
title(['Cell ' num2str(sp)])
set(gca,'xtick',[],'ytick',[])
ylabel('Event Rate')
end

subplot(7,2,13)
hold on
h1 = area([t_on t_end],[1 1]);
h2 = area([t_on t_end],[-1 -1]);
h1(:).FaceColor = [.7 .7 .7];
h1(:).EdgeColor = 'none';
h1(:).FaceAlpha = .5;
h2(:).FaceColor = [.7 .7 .7];
h2(:).EdgeColor = 'none';
h2(:).FaceAlpha = .5;
plot(f,'b')
plot(Wo'*r_init,'r')
plot([t_on t_on],[-1 1],'-k')
plot([t_off t_off],[-1 1],'-k')
ylim([-1 1])
title('Output')
set(gca,'xtick',[],'ytick',[])
ylabel('Event Rate')

%% training
for l = 1:n_train_its
x = randn(n_all,1);
r = tanh(x);
for t = 1:t_end
    r_train(:,t) = r;
    
    %innate training
    ei(l,:,t) = r_train(:,t) - R_all(:,t);    
    
    for i = 1:n_all
        P = Pi_cell{i};
        Pi_cell{i} = P - (P*r*r'*P)/(1+r'*P*r);
        Wr(:,i) = Wr(:,i) - ei(l,i,t) * Pi_cell{i}*r;
        
        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,:)*y(t,:)' + I_noise(i)); %+ Wfb(i)*z
    end
    x = x+dx;
    r = tanh(x);   
    
    eo(l,:,t) = Wo'*r - f(t);    
    for n = 1:n_out
        Po = Po_cell{n};
        Po_cell{n} = Po - (Po*r*r'*Po)/(1+r'*Po*r);
        Wo(:,n) = Wo(:,n) - eo(l,n,t) * Po_cell{n}*r;
    end
end
end

%% post train output
for t = 1:n_its
    for i = 1:n_all
        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,:)*y(t,:)' + I_noise(i));
    end
    x = x+dx;
    r = tanh(x);
    z = Wo'*r;
    r_test(:,t) = r;
end

for sp = 1:3
subplot(7,2,6+2*sp)
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
plot([t_off t_off],[-1 1],'-k')
title(['Cell ' num2str(sp)])
set(gca,'xtick',[],'ytick',[])
ylabel('Event Rate')
end

subplot(7,2,14)
hold on
h1 = area([t_on t_end],[1 1]);
h2 = area([t_on t_end],[-1 -1]);
h1(:).FaceColor = [.7 .7 .7];
h1(:).EdgeColor = 'none';
h1(:).FaceAlpha = .5;
h2(:).FaceColor = [.7 .7 .7];
h2(:).EdgeColor = 'none';
h2(:).FaceAlpha = .5;
plot(f,'b')
plot(Wo'*r_test,'r')
plot([t_on t_on],[-1 1],'-k')
plot([t_off t_off],[-1 1],'-k')
ylim([-1 1])
title('Output')
set(gca,'xtick',[],'ytick',[])
ylabel('Event Rate')