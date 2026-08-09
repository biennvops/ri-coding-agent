use std::io::{self, Stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use anyhow::Result;
use crossterm::cursor::Show;
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

static TERMINAL_MODE_ACTIVE: AtomicBool = AtomicBool::new(false);
static PANIC_HOOK: Once = Once::new();

pub struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    mode: TerminalModeGuard<CrosstermModeOps>,
}

pub(crate) trait TerminalModeOps {
    fn enable_raw(&mut self) -> io::Result<()>;
    fn enter_alternate(&mut self) -> io::Result<()>;
    fn leave_alternate(&mut self) -> io::Result<()>;
    fn disable_raw(&mut self) -> io::Result<()>;
    fn show_cursor(&mut self) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
}

#[derive(Default)]
struct CrosstermModeOps;

impl TerminalModeOps for CrosstermModeOps {
    fn enable_raw(&mut self) -> io::Result<()> {
        enable_raw_mode()
    }

    fn enter_alternate(&mut self) -> io::Result<()> {
        execute!(io::stdout(), EnterAlternateScreen)
    }

    fn leave_alternate(&mut self) -> io::Result<()> {
        execute!(io::stdout(), LeaveAlternateScreen)
    }

    fn disable_raw(&mut self) -> io::Result<()> {
        disable_raw_mode()
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        execute!(io::stdout(), Show)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

struct TerminalModeGuard<O: TerminalModeOps> {
    ops: O,
    raw_enabled: bool,
    alternate_screen: bool,
}

impl<O> TerminalModeGuard<O>
where
    O: TerminalModeOps,
{
    fn new(ops: O) -> Self {
        Self {
            ops,
            raw_enabled: false,
            alternate_screen: false,
        }
    }

    fn enable_raw(&mut self) -> io::Result<()> {
        self.ops.enable_raw()?;
        self.raw_enabled = true;
        TERMINAL_MODE_ACTIVE.store(true, Ordering::Release);
        Ok(())
    }

    fn enter_alternate(&mut self) -> io::Result<()> {
        self.ops.enter_alternate()?;
        self.alternate_screen = true;
        TERMINAL_MODE_ACTIVE.store(true, Ordering::Release);
        Ok(())
    }

    fn restore(&mut self) -> io::Result<()> {
        if !self.raw_enabled && !self.alternate_screen {
            return Ok(());
        }

        // A panic hook may already have restored the process-global terminal
        // state. Mark this guard inactive without issuing a second transition.
        if !TERMINAL_MODE_ACTIVE.load(Ordering::Acquire) {
            self.raw_enabled = false;
            self.alternate_screen = false;
            return Ok(());
        }

        let mut first_error = None;
        record_cleanup(&mut first_error, self.ops.show_cursor());
        if self.alternate_screen {
            record_cleanup(&mut first_error, self.ops.leave_alternate());
        }
        if self.raw_enabled {
            record_cleanup(&mut first_error, self.ops.disable_raw());
        }
        record_cleanup(&mut first_error, self.ops.flush());

        self.raw_enabled = false;
        self.alternate_screen = false;
        TERMINAL_MODE_ACTIVE.store(false, Ordering::Release);
        first_error.map_or(Ok(()), Err)
    }
}

impl<O> Drop for TerminalModeGuard<O>
where
    O: TerminalModeOps,
{
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

fn record_cleanup(first_error: &mut Option<io::Error>, result: io::Result<()>) {
    if let Err(error) = result {
        if first_error.is_none() {
            *first_error = Some(error);
        }
    }
}

fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            emergency_restore();
            previous(panic);
        }));
    });
}

fn emergency_restore() {
    if !TERMINAL_MODE_ACTIVE.swap(false, Ordering::AcqRel) {
        return;
    }

    // Each operation is independent so one failed terminal transition cannot
    // prevent the remaining emergency cleanup or the original panic report.
    let mut stdout = io::stdout();
    let _ = execute!(stdout, Show);
    let _ = execute!(stdout, LeaveAlternateScreen);
    let _ = disable_raw_mode();
    let _ = stdout.flush();
}

impl TerminalGuard {
    pub fn new() -> Result<Self> {
        install_panic_hook();
        let mut mode = TerminalModeGuard::new(CrosstermModeOps);
        mode.enable_raw()?;
        mode.enter_alternate()?;

        let stdout = io::stdout();
        let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;
        if let Err(error) = terminal.clear() {
            let _ = mode.restore();
            return Err(error.into());
        }

        Ok(Self { terminal, mode })
    }

    pub fn restore(&mut self) -> io::Result<()> {
        self.mode.restore()
    }

    pub fn terminal_mut(&mut self) -> &mut Terminal<CrosstermBackend<Stdout>> {
        &mut self.terminal
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[derive(Default)]
    struct FakeModeOps {
        calls: Vec<&'static str>,
        failures: HashSet<&'static str>,
    }

    impl FakeModeOps {
        fn fail_on(mut self, operation: &'static str) -> Self {
            self.failures.insert(operation);
            self
        }

        fn call(&mut self, operation: &'static str) -> io::Result<()> {
            self.calls.push(operation);
            if self.failures.contains(operation) {
                Err(io::Error::other(format!("fake {operation} failure")))
            } else {
                Ok(())
            }
        }
    }

    impl TerminalModeOps for FakeModeOps {
        fn enable_raw(&mut self) -> io::Result<()> {
            self.call("enable_raw")
        }

        fn enter_alternate(&mut self) -> io::Result<()> {
            self.call("enter_alternate")
        }

        fn leave_alternate(&mut self) -> io::Result<()> {
            self.call("leave_alternate")
        }

        fn disable_raw(&mut self) -> io::Result<()> {
            self.call("disable_raw")
        }

        fn show_cursor(&mut self) -> io::Result<()> {
            self.call("show_cursor")
        }

        fn flush(&mut self) -> io::Result<()> {
            self.call("flush")
        }
    }

    fn active_mode() -> TerminalModeGuard<FakeModeOps> {
        let mut guard = TerminalModeGuard::new(FakeModeOps::default());
        guard.enable_raw().unwrap();
        guard.enter_alternate().unwrap();
        guard
    }

    #[test]
    fn restore_is_ordered_and_drop_is_safe() {
        let mut guard = active_mode();
        guard.restore().unwrap();
        assert_eq!(
            guard.ops.calls,
            vec![
                "enable_raw",
                "enter_alternate",
                "show_cursor",
                "leave_alternate",
                "disable_raw",
                "flush"
            ]
        );
        drop(guard);
    }

    #[test]
    fn restore_then_drop_does_not_repeat_transitions() {
        let mut guard = active_mode();
        guard.restore().unwrap();
        let calls = guard.ops.calls.len();
        guard.restore().unwrap();
        assert_eq!(guard.ops.calls.len(), calls);
    }

    #[test]
    fn restoration_attempts_remaining_steps_after_an_error() {
        let mut guard = TerminalModeGuard::new(FakeModeOps::default().fail_on("leave_alternate"));
        guard.enable_raw().unwrap();
        guard.enter_alternate().unwrap();

        assert!(guard.restore().is_err());
        assert_eq!(
            guard.ops.calls,
            vec![
                "enable_raw",
                "enter_alternate",
                "show_cursor",
                "leave_alternate",
                "disable_raw",
                "flush"
            ]
        );
    }

    #[test]
    fn failed_raw_enable_does_not_claim_terminal_ownership() {
        let mut guard = TerminalModeGuard::new(FakeModeOps::default().fail_on("enable_raw"));
        assert!(guard.enable_raw().is_err());
        assert!(guard.restore().is_ok());
        assert_eq!(guard.ops.calls, vec!["enable_raw"]);
    }

    #[test]
    fn failed_alternate_entry_restores_raw_mode() {
        let mut guard = TerminalModeGuard::new(FakeModeOps::default().fail_on("enter_alternate"));
        guard.enable_raw().unwrap();
        assert!(guard.enter_alternate().is_err());
        assert!(guard.restore().is_ok());
        assert_eq!(
            guard.ops.calls,
            vec![
                "enable_raw",
                "enter_alternate",
                "show_cursor",
                "disable_raw",
                "flush"
            ]
        );
    }

    #[test]
    fn inactive_panic_restore_is_a_no_op() {
        TERMINAL_MODE_ACTIVE.store(false, Ordering::Release);
        emergency_restore();
        assert!(!TERMINAL_MODE_ACTIVE.load(Ordering::Acquire));
    }
}
