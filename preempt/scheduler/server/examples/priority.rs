#![feature(let_chains)]

use std::collections::{BTreeSet, HashMap};

use server::{Policy, Scheduler};

#[derive(Debug, Default)]
struct PriorityPolicy {
    cur_ctx: Option<u32>,
    pending_ctx: BTreeSet<(i32, u32)>,
    stopped_ctx: BTreeSet<u32>,
    // lower value means higher priority
    priorities: HashMap<u32, i32>,
}

impl PriorityPolicy {
    fn resume_ctx(&mut self, pid: u32, scheduler: &Scheduler) {
        if self.stopped_ctx.remove(&pid) {
            scheduler.resume_ctx(pid).ok();
        }
    }

    fn stop_ctx(&mut self, pid: u32, scheduler: &Scheduler) {
        if self.stopped_ctx.insert(pid) {
            scheduler.stop_ctx(pid).ok();
        }
    }
}

impl Policy for PriorityPolicy {
    fn on_ctx_create(&mut self, pid: u32, _scheduler: &Scheduler) {
        self.priorities.entry(pid).or_insert(0);
    }

    fn on_ctx_ready(&mut self, pid: u32, scheduler: &Scheduler) {
        if let Some(cur_ctx) = self.cur_ctx {
            // Keep the highest priority context running and stop the other one
            let ctx_to_stop = if self.priorities[&pid] < self.priorities[&cur_ctx] {
                self.cur_ctx = Some(pid);
                cur_ctx
            } else {
                pid
            };
            self.stop_ctx(ctx_to_stop, scheduler);
            let priority = self.priorities[&ctx_to_stop];
            self.pending_ctx.insert((priority, ctx_to_stop));
        } else {
            self.cur_ctx = Some(pid);
            self.resume_ctx(pid, scheduler);
        }
    }

    fn on_ctx_idle(&mut self, pid: u32, scheduler: &Scheduler) {
        if self.cur_ctx == Some(pid) {
            if let Some((_, next_ctx)) = self.pending_ctx.pop_first() {
                self.resume_ctx(next_ctx, scheduler);
                self.cur_ctx = Some(next_ctx);
            } else {
                self.cur_ctx = None;
            }
        } else {
            self.pending_ctx.retain(|&(_, p)| p != pid);
        }
    }

    fn on_ctx_release(&mut self, pid: u32, _scheduler: &Scheduler) {
        self.priorities.remove(&pid);
    }

    fn on_set_priority(&mut self, pid: u32, priority: i32, scheduler: &Scheduler) {
        self.priorities.insert(pid, priority);
        // if current context's priority is lower than pending context's, switch
        if let Some(cur_ctx) = self.cur_ctx
            && let Some(&(_, pending_ctx)) = self.pending_ctx.first() 
            && let priority = self.priorities[&cur_ctx]
            && priority < self.priorities[&pending_ctx]
        {
            self.stop_ctx(cur_ctx, scheduler);
            self.resume_ctx(pending_ctx, scheduler);
            self.cur_ctx = Some(pending_ctx);
            self.pending_ctx.pop_first();
            self.pending_ctx.insert((priority, cur_ctx));
        }
    }

    fn get_priority(&self, pid: u32) -> i32 {
        self.priorities.get(&pid).copied().unwrap_or(0)
    }
}

fn main() {
    env_logger::init();

    let policy = PriorityPolicy::default();
    server::enter_server(policy);
}
