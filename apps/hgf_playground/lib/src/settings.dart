import 'dart:math' as math;

import 'package:easy_shared_preferences/easy_shared_preferences.dart';
import 'package:material_ui/material_ui.dart';

import 'model/fit_config.dart';

/// App preferences kept between sessions: the theme (following the system
/// by default), the last preset and the number of trials, and how fits run
/// ([fit]).
class Settings extends ChangeNotifier {
  EasySettings? _store;

  ThemeMode themeMode = ThemeMode.system;
  String preset = '';
  int nTrials = 0;
  FitConfig fit = const FitConfig();

  static const _theme = 'app.theme';
  static const _preset = 'app.preset';
  static const _nTrials = 'app.n_trials';
  static const _device = 'fit.device';
  static const _workers = 'fit.workers';
  static const _restarts = 'fit.restarts';
  static const _chains = 'fit.chains';
  static const _warmup = 'fit.warmup';
  static const _draws = 'fit.draws';
  static const _targetAccept = 'fit.target_accept';
  static const _maxTreeDepth = 'fit.max_tree_depth';

  /// Read the saved values; without storage (or in tests) the defaults
  /// stay.
  Future<void> load() async {
    try {
      final store = SettingsStore();
      const d = FitConfig();
      final settings = EasySettings(store: store)
        ..register(
          SettingsGroup(
            key: 'app',
            store: store,
            items: [
              StringSetting(
                key: 'theme',
                defaultValue: ThemeMode.system.name,
                validator: EnumValidator<String>([
                  for (final m in ThemeMode.values) m.name,
                ]).validate,
              ),
              StringSetting(key: 'preset', defaultValue: ''),
              IntSetting(key: 'n_trials', defaultValue: 0),
            ],
          ),
        )
        ..register(
          SettingsGroup(
            key: 'fit',
            store: store,
            items: [
              StringSetting(
                key: 'device',
                defaultValue: d.device.name,
                validator: EnumValidator<String>([
                  for (final v in FitDevice.values) v.name,
                ]).validate,
              ),
              IntSetting(key: 'workers', defaultValue: d.workers),
              IntSetting(key: 'restarts', defaultValue: d.restarts),
              IntSetting(key: 'chains', defaultValue: d.chains),
              IntSetting(key: 'warmup', defaultValue: d.warmup),
              IntSetting(key: 'draws', defaultValue: d.draws),
              DoubleSetting(key: 'target_accept', defaultValue: d.targetAccept),
              IntSetting(key: 'max_tree_depth', defaultValue: d.maxTreeDepth),
            ],
          ),
        );
      await settings.init();
      _store = settings;
      themeMode = ThemeMode.values.byName(settings.getString(_theme));
      preset = settings.getString(_preset);
      nTrials = settings.getInt(_nTrials);
      int positive(String key, int fallback) {
        final v = settings.getInt(key);
        return v > 0 ? v : fallback;
      }

      fit = FitConfig(
        device: FitDevice.values.byName(settings.getString(_device)),
        workers: math.max(0, settings.getInt(_workers)),
        restarts: math.max(0, settings.getInt(_restarts)),
        chains: positive(_chains, d.chains),
        warmup: positive(_warmup, d.warmup),
        draws: positive(_draws, d.draws),
        targetAccept: switch (settings.getDouble(_targetAccept)) {
          final v when v > 0 && v < 1 => v,
          _ => d.targetAccept,
        },
        maxTreeDepth: positive(_maxTreeDepth, d.maxTreeDepth),
      );
    } catch (_) {
      // No storage available: keep the defaults.
    }
    notifyListeners();
  }

  void setThemeMode(ThemeMode mode) {
    if (mode == themeMode) return;
    themeMode = mode;
    notifyListeners();
    _save(_theme, mode.name);
  }

  /// Change how fits run; saves the values that changed.
  void setFit(FitConfig c) {
    final old = fit;
    if (c == old) return;
    fit = c;
    notifyListeners();
    if (c.device != old.device) _save(_device, c.device.name);
    if (c.workers != old.workers) _save(_workers, c.workers);
    if (c.restarts != old.restarts) _save(_restarts, c.restarts);
    if (c.chains != old.chains) _save(_chains, c.chains);
    if (c.warmup != old.warmup) _save(_warmup, c.warmup);
    if (c.draws != old.draws) _save(_draws, c.draws);
    if (c.targetAccept != old.targetAccept) {
      _save(_targetAccept, c.targetAccept);
    }
    if (c.maxTreeDepth != old.maxTreeDepth) {
      _save(_maxTreeDepth, c.maxTreeDepth);
    }
  }

  void remember({String? preset, int? nTrials}) {
    if (preset != null && preset != this.preset) {
      this.preset = preset;
      _save(_preset, preset);
    }
    if (nTrials != null && nTrials != this.nTrials) {
      this.nTrials = nTrials;
      _save(_nTrials, nTrials);
    }
  }

  void _save(String key, Object value) {
    final s = _store;
    if (s == null) return;
    // Fire and forget: a failed write only loses the preference.
    s.setValue(key, value).catchError((Object _) {});
  }
}
