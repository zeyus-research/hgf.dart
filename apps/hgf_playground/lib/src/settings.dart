import 'package:easy_shared_preferences/easy_shared_preferences.dart';
import 'package:flutter/material.dart';

/// App preferences kept between sessions: the theme (following the system
/// by default), the last preset and the number of trials.
class Settings extends ChangeNotifier {
  EasySettings? _store;

  ThemeMode themeMode = ThemeMode.system;
  String preset = '';
  int nTrials = 0;

  static const _theme = 'app.theme';
  static const _preset = 'app.preset';
  static const _nTrials = 'app.n_trials';

  /// Read the saved values; without storage (or in tests) the defaults
  /// stay.
  Future<void> load() async {
    try {
      final store = SettingsStore();
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
        );
      await settings.init();
      _store = settings;
      themeMode = ThemeMode.values.byName(settings.getString(_theme));
      preset = settings.getString(_preset);
      nTrials = settings.getInt(_nTrials);
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
