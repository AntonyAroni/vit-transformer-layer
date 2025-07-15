#include "../../include/training/lr_scheduler.h"
#include <cmath>
#include <algorithm>

LRScheduler::LRScheduler(double lr, int warmup, int total) 
    : initial_lr(lr), current_lr(lr), warmup_steps(warmup), total_steps(total), current_step(0) {}

double LRScheduler::getNextLR() {
    current_step++;
    
    if (current_step <= warmup_steps) {
        // Linear warmup
        current_lr = initial_lr * current_step / warmup_steps;
    } else {
        // Cosine annealing
        double progress = (double)(current_step - warmup_steps) / (total_steps - warmup_steps);
        current_lr = initial_lr * 0.5 * (1 + cos(M_PI * progress));
    }
    
    return current_lr;
}