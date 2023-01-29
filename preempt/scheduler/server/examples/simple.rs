use std::collections::VecDeque;

use server::{Policy, Scheduler};

#[derive(Debug, Default)]
struct DummyPolicy {
    cur_ctx: Option<u32>,
    pending_ctx: VecDeque<u32>,
}

impl Policy for DummyPolicy {
    fn on_ctx_create(&mut self, _ctx_id: u32, _scheduler: &Scheduler) {}

    fn on_ctx_ready(&mut self, ctx_id: u32, scheduler: &Scheduler) {
        if self.cur_ctx.is_none() {
            self.cur_ctx = Some(ctx_id);
        } else {
            scheduler.stop_ctx(ctx_id).ok();
            self.pending_ctx.push_back(ctx_id);
        }
    }

    fn on_ctx_idle(&mut self, ctx_id: u32, scheduler: &Scheduler) {
        if self.cur_ctx == Some(ctx_id) {
            if let Some(next_ctx) = self.pending_ctx.pop_front() {
                scheduler.resume_ctx(next_ctx).ok();
                self.cur_ctx = Some(next_ctx);
            } else {
                self.cur_ctx = None;
            }
        } else {
            self.pending_ctx.retain(|&x| x != ctx_id);
        }
    }

    fn on_ctx_release(&mut self, _ctx_id: u32, _scheduler: &Scheduler) {}
}

fn main() {
    env_logger::init();

    let policy = DummyPolicy::default();
    server::enter_server(policy);
}
