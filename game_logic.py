import random
import numpy as np

random.seed(80085)


#SUITS = ['♠', '♣', '♦', '♥']
SUITS = ['♠'] * 4
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

CARD_EMB_LEN = len(SUITS) + len(RANKS) + 1

class Card:
    def __init__(self, suit, rank):
        self.name = suit + rank
        self.suit = suit
        self.rank = RANKS.index(rank)
        self.hidden = False

    def repr(self):
        res = np.zeros(CARD_EMB_LEN)
        res[-1] = int(self.hidden)
        if not self.hidden:
            res[SUITS.index(self.suit)] = 1
            res[len(SUITS) + self.rank] = 1
        return res


class Stack:
    def __init__(self):
        self.stack = []
        self.can_take_n = 1

    def can_add(self, st):
        return self.stack[-1].rank - st[0].rank == 1

    def add(self, st):
        self.stack += st

        if self.stack[-1].suit == st[0].suit:
            self.can_take_n += len(st)
        else:
            self.can_take_n = len(st)

    def can_take(self, n):
        return n <= self.can_take_n

    def check_row(self):
        if self.can_take_n == len(RANKS) and self.stack[-self.can_take_n].rank == len(RANKS)-1 and self.stack[-1].rank == 0:
            _ = take(self.can_take_n, take=True)
            return True
        return False

    def take(self, n, take=True):
        if not self.can_take(n): return None

        ret = self.stack[-n:]

        if take:
            self.stack = self.stack[:-n]
            self.can_take_n -= n

            if self.can_take_n == 0 and len(self.stack) > 0:
                self.stack[-1].hidden = False
                self.can_take_n = 1

        return ret

    def print(self):
        print(' '.join([x.name for x in self.stack]))


class Game:
    def __init__(self):
        cards = []
        for s in SUITS:
            for r in RANKS:
                cards.append({'suit': s, 'rank': r})

        cards = cards * 2
        random.shuffle(cards)

        clsd = cards[:50]
        cards = cards[50:]
        self.closed = [clsd[0::5], clsd[1::5], clsd[2::5], clsd[3::5], clsd[4::5]]

        self.stacks = [Stack() for _ in range(10)]
        for i, c in enumerate(cards):
            card = Card(c['suit'], c['rank'])
            card.hidden = i < len(cards)-10
            self.stacks[i % 10].stack.append(card)

    def make_move(self, frm, to, n_take):
        if self.stacks[frm].can_take(n_take):
            taken = self.stacks[frm].take(n_take, take=False)

            if self.stacks[to].can_add(taken):
                _ = self.stacks[frm].take(n_take, take=True)
                status = 'ok move'

                if taken[0].suit == self.stacks[to].stack[-1].suit: status = 'same suit move'

                self.stacks[to].add(taken)

                return status
        return 'wrong move'

    def unclose_one(self):
        if len(self.closed) == 0: return 'wrong move'

        min_stack = min([len(stack.stack) for stack in self.stacks])
        if min_stack == 0: return 'wrong move'

        closed = self.closed.pop()
        for stack in self.stacks:
            card = closed.pop()
            stack.stack.append(Card(card['suit'], card['rank']))

        return 'ok'

    def check_rows(self):
        r = 0
        for stack in self.stacks:
            r += int(stack.check_row())
        return r

    def print(self):
        for i, stack in enumerate(self.stacks):
            s = str(i) + '\t'
            for card in stack.stack:
                if card.hidden: s += '[]'
                else: s += card.name
            print(s)