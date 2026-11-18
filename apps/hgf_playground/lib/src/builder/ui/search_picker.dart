import 'package:material_ui/material_ui.dart';

/// Lists longer than this are picked from [showSearchPicker] (a search
/// field over the list) instead of a dropdown, which gets unwieldy with
/// many nodes, columns or parameters.
const int searchPickerAbove = 8;

/// [s] for matching what is typed: lower case, subscript digits as digits
/// (ids print as x₁₂, people type x12).
String searchKey(String s) {
  final out = StringBuffer();
  for (final r in s.toLowerCase().runes) {
    out.writeCharCode(r >= 0x2080 && r <= 0x2089 ? r - 0x2080 + 0x30 : r);
  }
  return out.toString();
}

/// Whether [label] matches [query]: every word of the query appears in it.
bool searchMatches(String label, String query) {
  final key = searchKey(label);
  return searchKey(
    query,
  ).split(RegExp(r'\s+')).where((w) => w.isNotEmpty).every(key.contains);
}

/// A dialog listing [items] (value → label) under a search field, [groupOf]
/// giving each a heading; [selected] is ticked. Returns the pick in a
/// record (so that a nullable value can be picked), or null when dismissed.
Future<(T,)?> showSearchPicker<T>(
  BuildContext context, {
  required String title,
  required Map<T, String> items,
  T? selected,
  String? Function(T)? groupOf,
}) => showDialog<(T,)>(
  context: context,
  builder: (context) => _SearchPicker<T>(
    title: title,
    items: items,
    selected: selected,
    groupOf: groupOf,
  ),
);

class _SearchPicker<T> extends StatefulWidget {
  const _SearchPicker({
    required this.title,
    required this.items,
    required this.selected,
    required this.groupOf,
  });

  final String title;
  final Map<T, String> items;
  final T? selected;
  final String? Function(T)? groupOf;

  @override
  State<_SearchPicker<T>> createState() => _SearchPickerState<T>();
}

class _SearchPickerState<T> extends State<_SearchPicker<T>> {
  String _query = '';

  List<MapEntry<T, String>> get _shown => [
    for (final e in widget.items.entries)
      if (searchMatches(
        '${e.value} ${widget.groupOf?.call(e.key) ?? ''}',
        _query,
      ))
        e,
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final shown = _shown;
    // On touch screens the keyboard would cover half the list: search on
    // demand there.
    final touch = switch (theme.platform) {
      TargetPlatform.android || TargetPlatform.iOS => true,
      _ => false,
    };
    final children = <Widget>[];
    String? group;
    for (final MapEntry(:key, :value) in shown) {
      final g = widget.groupOf?.call(key);
      if (g != null && g != group) {
        children.add(
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
            child: Text(
              g,
              style: theme.textTheme.labelMedium?.copyWith(
                color: theme.colorScheme.primary,
              ),
            ),
          ),
        );
      }
      group = g;
      final isSelected = key == widget.selected;
      children.add(
        ListTile(
          dense: true,
          selected: isSelected,
          title: Text(value),
          trailing: isSelected ? const Icon(Icons.check, size: 18) : null,
          onTap: () => Navigator.of(context).pop((key,)),
        ),
      );
    }
    return AlertDialog(
      title: Text(widget.title),
      contentPadding: const EdgeInsets.fromLTRB(0, 12, 0, 0),
      content: SizedBox(
        width: 420,
        height: 440,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: TextField(
                autofocus: !touch,
                decoration: InputDecoration(
                  prefixIcon: const Icon(Icons.search),
                  hintText: 'Search ${widget.items.length}',
                  isDense: true,
                  border: const OutlineInputBorder(),
                ),
                onChanged: (q) => setState(() => _query = q),
                onSubmitted: (_) {
                  if (shown.length == 1) {
                    Navigator.of(context).pop((shown.single.key,));
                  }
                },
              ),
            ),
            const SizedBox(height: 4),
            Expanded(
              child: shown.isEmpty
                  ? Center(
                      child: Text(
                        'Nothing matches "$_query"',
                        style: theme.textTheme.bodySmall,
                      ),
                    )
                  : ListView(children: children),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
      ],
    );
  }
}
