from mcp.server import MCPServer

from mcp_server.tools.sanctions_lookup import lookup_sanctions
from mcp_server.tools.transaction_pattern_check import check_transaction_pattern


mcp = MCPServer(
    name="ComplyAgent MCP Server",
    description="MCP tools for sanctions screening and AML transaction-pattern analysis.",
)


@mcp.tool()
def sanctions_lookup(
    full_name: str,
    date_of_birth: str = "",
    nationality: str = "",
    country: str = "",
) -> dict:
    """Check a customer identity against OFAC and UN sanctions data."""
    return lookup_sanctions(
        full_name=full_name,
        date_of_birth=date_of_birth,
        nationality=nationality,
        country=country,
    )


@mcp.tool()
def transaction_pattern_check(
    customer_id: str,
) -> dict:
    """Check the synthetic AML transaction pattern assigned to a customer."""
    return check_transaction_pattern(customer_id)


if __name__ == "__main__":
    mcp.run()