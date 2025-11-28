# Twitter Auto-Discovery Feature

## Overview
The Twitter scanner now supports automatic trending topic detection. Instead of manually specifying keywords to search, the system can automatically fetch current trending topics from Twitter and search tweets for those topics.

## Configuration

### In `trend_scanner_agent.py`

```python
# Set to True to automatically discover trending topics
TWITTER_AUTO_DISCOVER_KEYWORDS = True

# Manual keywords (only used if AUTO_DISCOVER is False)
TARGET_TWITTER_KEYWORDS = [
    # Add keywords/hashtags here if auto-discover is disabled
]

# Twitter scan type
TWITTER_SCAN_TYPE = 'both'  # 'user', 'trending', or 'both'
```

### In `.env` file

```bash
# Enable automatic trending topic discovery
TWITTER_AUTO_DISCOVER_KEYWORDS=True
```

## How It Works

1. **When `TWITTER_AUTO_DISCOVER_KEYWORDS=True`**:
   - The system calls `twitter_tool.get_trending_topics()` to fetch current trending topics from Twitter
   - By default, fetches top 5 trending topics
   - These topics replace the manual `TARGET_TWITTER_KEYWORDS` list
   - The scanner then searches tweets for each trending topic

2. **When `TWITTER_AUTO_DISCOVER_KEYWORDS=False`**:
   - Uses manual `TARGET_TWITTER_KEYWORDS` list
   - Searches tweets for the specified keywords/hashtags

## Implementation Details

### New Method: `TwitterScanTool.get_trending_topics(limit=10)`

```python
def get_trending_topics(self, limit: int = 10) -> List[str]:
    """
    Fetch trending topics from Twitter
    
    Args:
        limit: Maximum number of trending topics to return
        
    Returns:
        List of trending topic names/hashtags
    """
```

### Updated Orchestration Flow

1. **GoogleAgentsManager.create_multi_platform_workflow()**:
   - New parameter: `twitter_auto_discover`
   - If `True`, calls `get_trending_topics()` before creating Twitter scan tasks
   - Auto-discovered topics replace manual keywords

2. **TrendScannerOrchestrator**:
   - New attribute: `twitter_auto_discover`
   - New parameter in `set_target_twitter()`: `auto_discover`
   - Passes flag through to workflow creation

## Usage Examples

### Example 1: Auto-Discover + User Accounts

```python
orchestrator.set_target_twitter(
    accounts=['BBCBreaking', 'CNN'],
    keywords=[],  # Ignored when auto_discover=True
    scan_type='both',
    auto_discover=True
)
```

This will:
- Scan tweets from @BBCBreaking and @CNN
- Auto-discover trending topics
- Search tweets for those trending topics

### Example 2: Manual Keywords Only

```python
orchestrator.set_target_twitter(
    accounts=[],
    keywords=['#election2024', 'breaking news'],
    scan_type='trending',
    auto_discover=False
)
```

This will:
- Search tweets for specified keywords only

### Example 3: Auto-Discover Only

```python
orchestrator.set_target_twitter(
    accounts=[],
    keywords=[],
    scan_type='trending',
    auto_discover=True
)
```

This will:
- Auto-discover trending topics
- Search tweets for those topics only

## Benefits

1. **Dynamic Detection**: Automatically adapts to current events and trending topics
2. **No Manual Updates**: No need to constantly update keyword lists
3. **Comprehensive Coverage**: Catches emerging misinformation narratives early
4. **Real-Time Relevance**: Always searches for what's currently trending

## API Details

The `get_trending_topics()` method uses Twikit's trending API:
- `client.get_trends('trending')` fetches current trending topics
- Returns trend objects with `name` attribute
- Extracts topic names into a list

## Error Handling

If auto-discovery fails:
- Logs error message
- Falls back to manual `TARGET_TWITTER_KEYWORDS` list
- Continues scanning with available targets

## Logging

The system logs auto-discovery activity:
```
INFO - Auto-discovering Twitter trending topics...
INFO - Auto-discovered 5 trending topics: ['Topic1', 'Topic2', ...]
WARNING - No trending topics auto-discovered, using manual keywords
ERROR - Failed to auto-discover trending topics: [error message]
```

## Configuration Priority

When `TWITTER_AUTO_DISCOVER_KEYWORDS=True`:
1. Auto-discovered topics take priority
2. Manual keywords in `TARGET_TWITTER_KEYWORDS` are ignored
3. User accounts in `TARGET_TWITTER_ACCOUNTS` are still scanned

## Complete Configuration Example

```python
# trend_scanner_agent.py

# User accounts to scan
TARGET_TWITTER_ACCOUNTS = [
    'BBCBreaking',
    'CNN',
    'Reuters'
]

# Auto-discover trending keywords
TWITTER_AUTO_DISCOVER_KEYWORDS = True

# Manual keywords (ignored when auto-discover is True)
TARGET_TWITTER_KEYWORDS = []

# Scan both user accounts and trending topics
TWITTER_SCAN_TYPE = 'both'
```

This configuration will:
1. Scan tweets from @BBCBreaking, @CNN, @Reuters
2. Auto-discover 5 trending topics from Twitter
3. Search tweets for those trending topics
4. Process all results through batch risk assessment
